"""
🌙 Moon Dev's Qwen (通义千问) Model Implementation
Built with love by Moon Dev 🚀

Supports Qwen series models via OpenAI-compatible API (DashScope)
"""

from openai import OpenAI
from termcolor import cprint
from .base_model import BaseModel, ModelResponse


class QwenModel(BaseModel):
    """Implementation for Alibaba's Qwen models (通义千问)"""

    AVAILABLE_MODELS = {
        "qwen-turbo": {
            "description": "Qwen Turbo - 高性价比模型，适合日常任务",
            "input_price": "¥0.002/1K tokens",
            "output_price": "¥0.006/1K tokens",
            "context_length": 131072
        },
        "qwen-plus": {
            "description": "Qwen Plus - 平衡性能与成本",
            "input_price": "¥0.004/1K tokens",
            "output_price": "¥0.012/1K tokens",
            "context_length": 131072
        },
        "qwen-max": {
            "description": "Qwen Max - 旗舰模型，适合复杂任务",
            "input_price": "¥0.02/1K tokens",
            "output_price": "¥0.06/1K tokens",
            "context_length": 32768
        },
        "qwen-max-longcontext": {
            "description": "Qwen Max Long - 超长上下文旗舰模型",
            "input_price": "¥0.02/1K tokens",
            "output_price": "¥0.06/1K tokens",
            "context_length": 30720
        },
        "qwen-long": {
            "description": "Qwen Long - 超长上下文模型，支持10M tokens",
            "input_price": "¥0.0005/1K tokens",
            "output_price": "¥0.002/1K tokens",
            "context_length": 10000000
        },
        "qwen-vl-max": {
            "description": "Qwen VL Max - 多模态旗舰，支持图像理解",
            "input_price": "¥0.02/1K tokens",
            "output_price": "¥0.06/1K tokens",
            "context_length": 32768
        },
        "qwen-vl-plus": {
            "description": "Qwen VL Plus - 多模态模型，高性价比",
            "input_price": "¥0.008/1K tokens",
            "output_price": "¥0.024/1K tokens",
            "context_length": 8192
        },
        "qwen-coder-turbo": {
            "description": "Qwen Coder Turbo - 代码专用模型，快速高效",
            "input_price": "¥0.002/1K tokens",
            "output_price": "¥0.006/1K tokens",
            "context_length": 131072
        },
        "qwen-coder-plus": {
            "description": "Qwen Coder Plus - 代码专用模型，性能更强",
            "input_price": "¥0.0035/1K tokens",
            "output_price": "¥0.007/1K tokens",
            "context_length": 131072
        },
        "qwen2.5-72b-instruct": {
            "description": "Qwen 2.5 72B - 开源最强模型",
            "input_price": "¥0.004/1K tokens",
            "output_price": "¥0.012/1K tokens",
            "context_length": 131072
        },
        "qwen2.5-32b-instruct": {
            "description": "Qwen 2.5 32B - 高性能开源模型",
            "input_price": "¥0.0035/1K tokens",
            "output_price": "¥0.007/1K tokens",
            "context_length": 131072
        },
        "qwen2.5-14b-instruct": {
            "description": "Qwen 2.5 14B - 平衡性能与效率",
            "input_price": "¥0.002/1K tokens",
            "output_price": "¥0.006/1K tokens",
            "context_length": 131072
        },
        "qwen2.5-7b-instruct": {
            "description": "Qwen 2.5 7B - 轻量高效模型",
            "input_price": "¥0.001/1K tokens",
            "output_price": "¥0.002/1K tokens",
            "context_length": 131072
        },
        "qwq-32b": {
            "description": "QwQ 32B - 推理增强模型，深度思考",
            "input_price": "¥0.0035/1K tokens",
            "output_price": "¥0.007/1K tokens",
            "context_length": 131072
        }
    }

    # DashScope API base URL (OpenAI compatible)
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str, model_name: str = "qwen-turbo", **kwargs):
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 4096)
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the Qwen client via OpenAI-compatible API"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL
            )
            cprint(f"✨ Moon Dev's magic initialized Qwen model: {self.model_name} 🌟", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize Qwen model: {str(e)}", "red")
            self.client = None

    def generate_response(self, system_prompt, user_content, **kwargs):
        """Generate a response using the Qwen model"""
        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            cprint(f"🤔 Moon Dev's {self.model_name} is thinking...", "yellow")

            # QwQ model requires enable_thinking for reasoning
            extra_kwargs = {}
            if self.model_name.startswith("qwq"):
                extra_kwargs["extra_body"] = {"enable_thinking": True}

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_kwargs
            )

            content = response.choices[0].message.content

            return ModelResponse(
                content=content.strip() if content else "",
                raw_response=response,
                model_name=self.model_name,
                usage=response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None
            )

        except Exception as e:
            cprint(f"❌ Qwen generation error: {repr(e)}", "red")
            raise

    def is_available(self) -> bool:
        """Check if Qwen is available"""
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "qwen"
