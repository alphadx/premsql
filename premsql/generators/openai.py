import os
from typing import Optional

from premsql.generators.base import Text2SQLGeneratorBase

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Module openai is not installed")


class Text2SQLGeneratorOpenAI(Text2SQLGeneratorBase):
    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        type: str,
        experiment_folder: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self._api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        super().__init__(
            experiment_folder=experiment_folder,
            experiment_name=experiment_name,
            type=type,
        )

    @property
    def load_client(self):
        client = OpenAI(api_key=self._api_key)
        return client

    @property
    def load_tokenizer(self):
        pass

    @property
    def model_name_or_path(self):
        return self.model_name

    @property
    def _uses_completion_tokens(self) -> bool:
        # GPT-5 family expects `max_completion_tokens` instead of `max_tokens`.
        return self.model_name.lower().startswith("gpt-5")

    def generate(
        self,
        data_blob: dict,
        temperature: Optional[float] = 0.0,
        max_new_tokens: Optional[int] = 256,
        postprocess: Optional[bool] = True,
        **kwargs
    ) -> str:
        prompt = data_blob["prompt"]
        token_key = "max_completion_tokens" if self._uses_completion_tokens else "max_tokens"
        token_budget = max_new_tokens if max_new_tokens is not None else 256
        if self._uses_completion_tokens:
            token_budget = max(token_budget, 1024)

        generation_config = {**kwargs, **{token_key: token_budget}}
        if not self._uses_completion_tokens:
            generation_config["temperature"] = temperature

        try:
            completion = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **generation_config
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            error_msg = str(e)
            max_token_error = "model output limit was reached" in error_msg or "max_tokens" in error_msg
            if self._uses_completion_tokens and max_token_error:
                retry_budget = min(token_budget * 2, 4096)
                generation_config[token_key] = retry_budget
                completion = (
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        **generation_config
                    )
                    .choices[0]
                    .message.content
                )
            else:
                raise

        return self.postprocess(output_string=completion) if postprocess else completion
