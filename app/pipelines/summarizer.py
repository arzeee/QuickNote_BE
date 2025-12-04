import whisper
import time

class Transcriber:
    """
    transcribe audio using OpenAI Whisper.
    """
    def __init__(self, model_name="small", gemini_model=None):
        print(f"load whisper model: '{model_name}'")
        start_time = time.time()
        self.model = whisper.load_model(model_name)
        self.gemini_model = gemini_model
        print(f"model successfully loaded in {time.time() - start_time:.2f} seconds.")

    def transcribe(self, file_path, language=None):
        print(f"start transcribing: {file_path}")
        if language:
            print(f"Forcing transcription in language: {language}")
        else:
            print("Language not specified, using auto-detection.")

        start_time = time.time()
        try:
            options = {"language": language} if language else {}
            result = self.model.transcribe(str(file_path),  **options) 
            
            print(f"transcribed {time.time() - start_time:.2f} seconds.")
            print(f"Detected language: {result['language']}")
            return result['text'], result['language']
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def language_rules(self, transcript, language):
        """
        Correct the transcript based on language rules
        """
        prompt = (
            f"You are a language expert. Please correct the following transcript "
            f"without changing the original meaning, just adjusting the language rules."
            f"according to the rules of the {language} transcription:\n\n{transcript}"
        )

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in language correction: {e}")
            return transcript

        
