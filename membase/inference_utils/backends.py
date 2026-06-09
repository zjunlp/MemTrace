from openai import OpenAI  
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import Any, Callable


class OpenAIClient(OpenAI): 
    """An OpenAI client with built-in retry and streaming support."""

    def get_text_generation_output(
        self, 
        messages: list[dict[str, Any]], 
        model: str = "gpt-4.1", 
        post_processor: Callable[[str], Any] | None = None,
        max_tolerance: int = 3,
        temperature: float | None = None, 
        top_p: float | None = None,
        stream: bool = True, 
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate text using the OpenAI chat completions API with retry and optional post-processing.

        Args:
            messages (`list[dict[str, Any]]`): 
                The input messages in OpenAI chat format.
            model (`str`, defaults to `"gpt-4.1"`): 
                The model identifier.
            post_processor (`Callable[[str], Any] | None`, optional): 
                An optional callable applied to the raw response content. 
                It is useful for parsing the response content, computing the cost or extracting related metadata.
            max_tolerance (`int`, defaults to `3`): 
                Maximum number of retry attempts.
            temperature (`float`, optional): 
                Sampling temperature.
            top_p (`float`, optional): 
                Nucleus sampling probability.
            stream (`bool`, defaults to `True`): 
                Whether to use streaming mode.
            **kwargs (`Any`): 
                Additional keyword arguments forwarded to the chat completions API.
                See https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
                for details.

        Returns:
            `dict[str, Any]`: 
                A dictionary containing raw text, post-processed result, and optionally 
                reasoning content if the model returns reasoning tokens.
        """
        response_content = None 
        counter = 0 
        content = ''
        reasoning_content = None 

        while response_content is None and counter <= max_tolerance:
            try:
                response = self.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream,
                    **kwargs
                )
                if stream:
                    chunks = []
                    reasoning_chunks = [] 
                    for chunk in response:
                        if len(chunk.choices) > 0:
                            if hasattr(chunk.choices[0].delta, "content"):
                                chunks.append(chunk.choices[0].delta.content or '')
                            if hasattr(chunk.choices[0].delta, "reasoning_content"):
                                reasoning_chunks.append(chunk.choices[0].delta.reasoning_content or '')
                        else:
                            warnings.warn(
                                "Find a chunk without `choices` attribute. "
                                "The model may reject to answer the question. "
                                "Please check the question and the model you use.",
                                UserWarning
                            )
                    content = ''.join(chunks)
                    if len(reasoning_chunks) > 0:
                        reasoning_content = ''.join(reasoning_chunks)
                else:
                    content = response.choices[0].message.content
                    if hasattr(response.choices[0].message, "reasoning_content"):
                        reasoning_content = response.choices[0].message.reasoning_content 
            except Exception as e:
                print(e)
            finally: 
                response_content = content if post_processor is None else post_processor(content)
                counter += 1
        
        outputs = {
            "content": content, 
            "processed_content": response_content,
        }
        if reasoning_content is not None:
            outputs["reasoning_content"] = reasoning_content

        return outputs


def openai_api_batch_inference(
    clients: list[OpenAIClient], 
    messages_list: list[list[dict[str, Any]]], 
    model: str = "gpt-4.1", 
    post_processor: Callable[[str], Any] | None = None,
    max_tolerance: int = 3,
    temperature: float = 0.75, 
    top_p: float = 0.95,
    stream: bool = True, 
    **kwargs: Any,
) -> list[dict[str, Any]]: 
    """Process multiple OpenAI API requests in parallel using a thread pool.

    Each client is paired with the corresponding message list at the same index. All pairs 
    are submitted concurrently via a thread pool.

    Args:
        clients (`list[OpenAIClient]`): 
            The OpenAI clients, one per request.
        messages_list (`list[list[dict[str, Any]]]`): 
            The messages for each client. Must have the same length as `clients`.
        model (`str`, defaults to `"gpt-4.1"`): 
            The model identifier.
        post_processor (`Callable[[str], Any] | None`, optional): 
            An optional callable applied to each raw response content.
        max_tolerance (`int`, defaults to `3`): 
            Maximum number of retry attempts per request.
        temperature (`float`, defaults to `0.75`): 
            Sampling temperature.
        top_p (`float`, defaults to `0.95`): 
            Nucleus sampling probability.
        stream (`bool`, defaults to `True`): 
            Whether to use streaming mode.
        **kwargs (`Any`): 
            Additional keyword arguments forwarded to each client's chat completion API.
            See https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
            for details.

    Returns:
        `list[dict[str, Any]]`: 
            A list of response dictionaries, one per client.
    """
    n_jobs = len(clients) 
    if len(messages_list) != n_jobs:
        raise ValueError(
            f"The number of clients ({n_jobs}) must match the number of messages ({len(messages_list)})."
        )
    
    apply_func = lambda client, messages: client.get_text_generation_output(
        messages, 
        model, 
        post_processor, 
        max_tolerance, 
        temperature, 
        top_p, 
        stream, 
        **kwargs, 
    )

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(apply_func, clients[i], messages_list[i])
            for i in range(n_jobs)
        ]
        results = [future.result() for future in futures]

    return results


class NativeLLMClient: 
    """A client for native large language model inference powered by vLLM."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize the client by loading a vLLM model and its tokenizer.

        Args:
            model (`str`): 
                The model name or path.
            **kwargs (`Any`): 
                Additional keyword arguments forwarded to ``vllm.LLM``.
        """
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Please install the required dependencies: "
                "`pip install transformers vllm`"
            ) from e
        
        self.model = LLM(model=model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def __call__(
        self, 
        messages_list: list[list[dict[str, str]]], 
        post_processor: Callable[[str], Any] | None = None, 
        enable_thinking: bool | None = None, 
        **kwargs: Any, 
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Generate text for one or more conversations using vLLM.

        Args:
            messages_list (`list[list[dict[str, str]]]`): 
                A list of conversations, each in OpenAI chat format.
            post_processor (`Callable[[str], Any] | None`, optional): 
                An optional callable applied to the raw response content. 
                It is useful for parsing the response content, computing the cost or extracting related metadata.
            enable_thinking (`bool | None`, optional): 
                It is used to enable or disable the model's thinking mode. It is only applicable to 
                some Qwen series models.
            **kwargs (`Any`): 
                Additional keyword arguments forwarded to `vllm.SamplingParams`. If no
                keyword arguments are provided, the default sampling params are used.

        Returns:
            `dict[str, Any] | list[dict[str, Any]]`: 
                A single response dictionary if there is only one conversation, or a list of 
                response dictionaries otherwise.
        """
        if enable_thinking is not None:
            texts = self.tokenizer.apply_chat_template(
                messages_list, 
                tokenize=False, 
                add_generation_prompt=True,  
                enable_thinking=enable_thinking, 
            )
        else:
            texts = self.tokenizer.apply_chat_template(
                messages_list, 
                tokenize=False, 
                # Add additional tokens to ensure the chat model
                # generate a system response instead of continuing a users message. 
                add_generation_prompt=True,  
            )
        
        if len(kwargs) > 0:
            from vllm import SamplingParams 
            sampling_params = SamplingParams(**kwargs)
            outputs = self.model.generate(texts, sampling_params)
        else: 
            # Use default sampling params recommended by the model creator. 
            outputs = self.model.generate(texts)

        new_outputs = [] 
        for output in outputs: 
            content = output.outputs[0].text 
            processed_content = content 
            if post_processor is not None:
                processed_content = post_processor(content)
            new_outputs.append(
                {
                    "content": content, 
                    "processed_content": processed_content, 
                }
            )

        return new_outputs if len(new_outputs) > 1 else new_outputs[0]
    

class OpenAIClientPool: 
    """A pool of OpenAI clients for batch inference."""

    def __init__(
        self, 
        api_keys: list[str] | str, 
        base_urls: list[str] | str, 
        model: str = "gpt-4.1", 
        **kwargs: Any,
    ) -> None:
        """Initialize the pool by creating one openai client per API key and base URL pair.

        Args:
            api_keys (`list[str] | str`): 
                API keys for the OpenAI clients.
            base_urls (`list[str] | str`): 
                Base URLs for the OpenAI clients. Must have the same length as `api_keys`.
            model (`str`, defaults to `"gpt-4.1"`): 
                The default model identifier used by all clients.
            **kwargs (`Any`): 
                Additional keyword arguments forwarded to each openai client.
        """
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        if isinstance(base_urls, str):
            base_urls = [base_urls]

        if len(api_keys) != len(base_urls):
            raise ValueError(
                f"The number of api keys ({len(api_keys)}) must match the number of base URLs ({len(base_urls)})."
            )
        
        self.client_pool = [
            OpenAIClient(
                api_key=api_key, 
                base_url=base_url, 
                **kwargs
            )
            for api_key, base_url in zip(api_keys, base_urls)
        ] 
        self.model = model  
    
    def __call__(
        self, 
        messages_list: list[list[dict[str, Any]]], 
        post_processor: Callable[[str], Any] | None = None, 
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Generate text for one or more conversations, automatically batched by pool size.

        Args:
            messages_list (`list[list[dict[str, Any]]]`): 
                A list of conversations, each in OpenAI chat format.
            post_processor (`Callable[[str], Any] | None`, optional): 
                An optional callable applied to each raw response content.
            **kwargs (`Any`): 
                Additional keyword arguments forwarded to each client's chat completion API.
                See https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
                for details.

        Returns:
            `dict[str, Any] | list[dict[str, Any]]`: 
                A single response dictionary if there is only one conversation, or a list of 
                response dictionaries otherwise.
        """
        max_batch_size = len(self.client_pool) 

        outputs = [] 
        for i in range(0, len(messages_list), max_batch_size):
            batch_messages_list = messages_list[i: i + max_batch_size]
            batch_clients = self.client_pool[0: len(batch_messages_list)]
            if len(batch_clients) == 1:
                client = batch_clients[0]
                outputs.append(
                    client.get_text_generation_output(
                        batch_messages_list[0], 
                        model=self.model, 
                        post_processor=post_processor, 
                        **kwargs
                    )
                ) 
            else:
                outputs.extend(
                    openai_api_batch_inference(
                        batch_clients, 
                        batch_messages_list, 
                        model=self.model, 
                        post_processor=post_processor, 
                        **kwargs
                    )
                )

        return outputs if len(outputs) > 1 else outputs[0]
    
    @property
    def pool_size(self) -> int:
        """The number of clients in the pool."""
        return len(self.client_pool)


def get_interface_for_inference(
    model: str, 
    api_keys: list[str] | str | None = None, 
    base_urls: list[str] | str | None = None, 
    **kwargs: Any,
) -> OpenAIClientPool | NativeLLMClient:
    """Create an inference interface based on the provided arguments.

    It returns a pool of openai clients when api keys and base URLs are both provided.
    Otherwise, it returns a client for local vLLM inference.

    Args:
        model (`str`): 
            The model name or path.
        api_keys (`list[str] | str | None`, optional): 
            API keys for OpenAI-compatible endpoints.
        base_urls (`list[str] | str | None`, optional): 
            Base URLs for OpenAI-compatible endpoints. Must be provided together with 
            `api_keys`.
        **kwargs (`Any`): 
            Additional keyword arguments forwarded to the underlying client constructor 
            (`OpenAIClientPool` or `NativeLLMClient`).

    Returns:
        `OpenAIClientPool | NativeLLMClient`: 
            The constructed inference interface.
    """
    if api_keys is not None and base_urls is not None:
        return OpenAIClientPool(
            api_keys, 
            base_urls, 
            model=model, 
            **kwargs,
        )
    if api_keys is not None or base_urls is not None:
        raise ValueError(
            "Either both `api_keys` and `base_urls` must be provided, or neither."
        )
    interface = NativeLLMClient(model, **kwargs) 
    return interface
