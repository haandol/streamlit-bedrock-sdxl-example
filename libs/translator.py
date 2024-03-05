import re
import boto3


class Translator(object):
    def __init__(self, profile_name: str, region_name: str):
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        self.pattern = re.compile(r"[^a-zA-Z0-9 ,.]+")
        self.client = session.client(service_name="translate")

    def translate(self, text: str, target_language: str = "en") -> str:
        if text is None:
            return ""

        # if text doescontains only English letters, return as is
        if not self.pattern.search(text):
            print("text is in English")
            return text

        L = []
        for t in text.split(","):
            response = self.client.translate_text(
                Text=t.strip(),
                SourceLanguageCode="auto",
                TargetLanguageCode=target_language,
            )
            L.append(response.get("TranslatedText", ""))
        return ", ".join(filter(None, L))
