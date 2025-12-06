from google import genai
from utils import load_json
import base64
import io
from PIL import Image


SYSTEM_PROMPT = (
    "explain easily with metaphors or examples, "
    "focus on the core idea, and always respond in the user's input language, "
    "regardless of previous conversation history."
)


class GeminiClient:

    def __init__(self):
        keydata = load_json("storage/api_key.json")
        if not keydata or "api_key" not in keydata:
            raise Exception("API Key not found.")

        self.client = genai.Client(api_key=keydata["api_key"])

        self.history = []
        self.max_history = 10


    def send_message(self, text="", image_b64=None, on_delta=None):

        # 히스토리 저장
        self.history.append({"text": text, "image_b64": image_b64})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]


        # ============================
        # contents 구성
        # ============================
        contents = []
        contents.append(SYSTEM_PROMPT)

        for item in self.history:

            # 텍스트
            if item["text"]:
                contents.append(item["text"])

            # 이미지 (★ 유일하게 허용되는 형식: PIL.Image)
            if item["image_b64"]:
                img_bytes = base64.b64decode(item["image_b64"])
                pil_img = Image.open(io.BytesIO(img_bytes))
                contents.append(pil_img)


        # ============================
        # 스트리밍 호출
        # ============================
        response = self.client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents
        )

        # ============================
        # 스트리밍 수신
        # ============================
        full_text = ""

        for chunk in response:
            if chunk.text:
                full_text += chunk.text
                if on_delta:
                    on_delta(chunk.text)

        # 히스토리에 추가
        self.history.append({"text": full_text, "image_b64": None})

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return full_text
