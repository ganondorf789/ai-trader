"""
🌙 Moon Dev's ZhipuAI (智谱AI) Model Implementation
Built with love by Moon Dev 🚀

Supports GLM-4 series models via zhipuai SDK
"""

from zhipuai import ZhipuAI
from termcolor import cprint
from .base_model import BaseModel, ModelResponse


class ZhipuModel(BaseModel):
    """Implementation for ZhipuAI's GLM models (智谱AI)"""

    AVAILABLE_MODELS = {
        "glm-4-plus": {
            "description": "GLM-4 Plus - 高智能旗舰模型，性能全面提升",
            "input_price": "¥0.05/1K tokens",
            "output_price": "¥0.05/1K tokens",
            "context_length": 128000
        },
        "glm-4-0520": {
            "description": "GLM-4 (0520) - 高智能模型，适合复杂任务",
            "input_price": "¥0.1/1K tokens",
            "output_price": "¥0.1/1K tokens",
            "context_length": 128000
        },
        "glm-4-air": {
            "description": "GLM-4 Air - 高性价比模型，推理快速",
            "input_price": "¥0.001/1K tokens",
            "output_price": "¥0.001/1K tokens",
            "context_length": 128000
        },
        "glm-4-airx": {
            "description": "GLM-4 AirX - 极速推理，适合实时场景",
            "input_price": "¥0.01/1K tokens",
            "output_price": "¥0.01/1K tokens",
            "context_length": 8192
        },
        "glm-4-flash": {
            "description": "GLM-4 Flash - 免费模型，适合轻量任务",
            "input_price": "Free",
            "output_price": "Free",
            "context_length": 128000
        },
        "glm-4-flashx": {
            "description": "GLM-4 FlashX - 免费极速模型",
            "input_price": "Free",
            "output_price": "Free",
            "context_length": 128000
        },
        "glm-4-long": {
            "description": "GLM-4 Long - 超长上下文，支持1M tokens",
            "input_price": "¥0.001/1K tokens",
            "output_price": "¥0.001/1K tokens",
            "context_length": 1000000
        },
        "glm-4v-plus": {
            "description": "GLM-4V Plus - 多模态旗舰，支持图像理解",
            "input_price": "¥0.01/1K tokens",
            "output_price": "¥0.01/1K tokens",
            "context_length": 8192
        },
        "glm-4v": {
            "description": "GLM-4V - 多模态模型，支持图像理解",
            "input_price": "¥0.05/1K tokens",
            "output_price": "¥0.05/1K tokens",
            "context_length": 2048
        }
    }

    def __init__(self, api_key: str, model_name: str = "glm-4-flash", **kwargs):
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 4096)
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the ZhipuAI client"""
        try:
            self.client = ZhipuAI(api_key=self.api_key)
            cprint(f"✨ Moon Dev's magic initialized ZhipuAI model: {self.model_name} 🌟", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize ZhipuAI model: {str(e)}", "red")
            self.client = None

    def generate_response(self, system_prompt, user_content, **kwargs):
        """Generate a response using the ZhipuAI model"""
        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            cprint(f"🤔 Moon Dev's {self.model_name} is thinking...", "yellow")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content

            return ModelResponse(
                content=content.strip() if content else "",
                raw_response=response,
                model_name=self.model_name,
                usage=response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None
            )

        except Exception as e:
            cprint(f"❌ ZhipuAI generation error: {repr(e)}", "red")
            raise

    def is_available(self) -> bool:
        """Check if ZhipuAI is available"""
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "zhipu"
