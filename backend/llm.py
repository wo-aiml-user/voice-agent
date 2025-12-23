"""
LLM module for Voice Agent.
Handles DeepSeek streaming responses.
"""

import time
from typing import AsyncGenerator, Callable, Optional

from client import deepseek_client, SYSTEM_PROMPT, logger


async def stream_llm_response(
    session_id: str,
    user_input: str,
    on_token: Optional[Callable[[str], None]] = None,
    on_first_token: Optional[Callable[[int], None]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens from DeepSeek.
    
    Args:
        session_id: Session identifier for logging
        user_input: User's transcribed input
        on_token: Optional callback for each token
        on_first_token: Optional callback with TTFT in milliseconds
        
    Yields:
        Response tokens as they arrive
    """
    try:
        llm_start_ts = time.perf_counter()
        first_token_logged = False
        
        logger.info(f"[{session_id}] LLM | Starting async stream...")
        
        # Use DeepSeek via AsyncOpenAI with async streaming
        response_stream = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            stream=True,
            temperature=0.4  # Lower temperature for faster, more consistent responses
        )
        
        full_response = ""
        
        # Async iteration over streaming response
        async for chunk in response_stream:
            # Extract text from OpenAI-style streaming response
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_response += text
                
                # Log first token (TTFT)
                if not first_token_logged:
                    ttft = int((time.perf_counter() - llm_start_ts) * 1000)
                    logger.info(f"[{session_id}] LLM | ⚡ First token (TTFT): {ttft}ms")
                    first_token_logged = True
                    if on_first_token:
                        on_first_token(ttft)
                
                # Call token callback if provided
                if on_token:
                    await on_token(text)
                
                yield text
        
        logger.info(f"[{session_id}] LLM | Response complete: '{full_response[:100]}...'")
        
    except Exception as e:
        logger.error(f"[{session_id}] LLM | Error in async stream: {e}")
        raise


async def get_llm_response_with_queue(
    session_id: str,
    user_input: str,
    token_queue,
    send_message_callback: Callable,
    is_active_check: Callable[[], bool],
    is_processing_check: Callable[[], bool],
) -> str:
    """
    Stream LLM response and push tokens to TTS queue.
    
    Args:
        session_id: Session identifier for logging
        user_input: User's transcribed input
        token_queue: Async queue for TTS tokens
        send_message_callback: Callback to send messages to frontend
        is_active_check: Function to check if session is active
        is_processing_check: Function to check if still processing
        
    Returns:
        Full response text
    """
    full_response = ""
    llm_start_ts = time.perf_counter()
    first_token_logged = False
    
    try:
        logger.info(f"[{session_id}] LLM | Starting async stream...")
        
        response_stream = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            stream=True,
            temperature=0.4
        )
        
        async for chunk in response_stream:
            if not is_active_check():
                break
            
            # Check for barge-in and stop processing if user interrupted
            if not is_processing_check():
                logger.info(f"[{session_id}] LLM | Stopping due to barge-in")
                break
            
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_response += text
                
                # Log first token (TTFT)
                if not first_token_logged:
                    ttft = int((time.perf_counter() - llm_start_ts) * 1000)
                    logger.info(f"[{session_id}] LLM | ⚡ First token (TTFT): {ttft}ms")
                    first_token_logged = True
                
                # Send token to frontend for live display
                await send_message_callback("llm_token", {"text": text})
                
                # Push token to TTS queue immediately
                await token_queue.put(text)
        
        # Signal TTS to flush remaining buffer
        await token_queue.put("[[FLUSH]]")
        
        logger.info(f"[{session_id}] LLM | Response complete: '{full_response[:100]}...'")
        
        # Send response complete signal
        await send_message_callback("llm_complete", {"text": full_response})
        
    except Exception as e:
        logger.error(f"[{session_id}] LLM | Error in async stream: {e}")
        await send_message_callback("error", {"message": f"LLM error: {str(e)}"})
        # Still need to flush TTS
        await token_queue.put("[[FLUSH]]")
    
    return full_response
