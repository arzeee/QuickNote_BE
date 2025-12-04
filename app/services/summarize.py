import os
from pathlib import Path
from dotenv import load_dotenv

from app.pipelines.converter import convert_video_to_audio, convert_audio_format
from app.pipelines.transcriber import Transcriber
from app.pipelines.summarizer import Summarizer
from app.pipelines.preprocessor import enhance_audio
from app.helper import JSONLogger
from app.pipelines.senopati_model import SenopatiModel  

load_dotenv()

try:
    print("Initializing Senopati Local Model...")
    senopati_model = SenopatiModel()
    print("Senopati model ready.")
except Exception as e:
    print(f"FATAL: Senopati model initialization failed: {e}")
    exit(1)


def run_pipeline(
    input_file: Path,
    denoise: bool = False,
    aggressive_denoise: bool = False,
    force_wav: bool = False,
    transcriber_model: str = "small",
    chunk_size: int = 2000,
    language: str = None
) -> dict:
    logger = JSONLogger()
    audio_path = None

    type_supported = [".mp4", ".mkv", ".mov", ".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    logger.log("INITIALIZATION", "INFO", "Starting audio/video processing pipeline", 
               input_file=str(input_file), 
               transcriber_model=transcriber_model,
               denoise_enabled=denoise,
               aggressive_denoise=aggressive_denoise)

    if not input_file.exists():
        logger.log("FILE_VALIDATION", "ERROR", f"Input file not found: {input_file}")
        return {"error": f"File not found: {input_file}"}
    
    input_type = input_file.suffix.lower()
    if input_type not in type_supported:
        logger.log("FILE_VALIDATION", "ERROR", f"Unsupported file type: {input_type}")
        return {"error": f"Unsupported file type: {input_type}"}

    logger.log("FILE_VALIDATION", "SUCCESS", "Input file validated", file_type=input_type)
    logger.log("AUDIO_CONVERSION", "INFO", "Starting audio format standardization")
    
    if input_type in [".mp4", ".mkv", ".mov"]:
        logger.log("VIDEO_PROCESSING", "INFO", f"Processing video file: '{input_file.name}'")
        output_dir = Path("./data/audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"{input_file.stem}.wav"
        result = convert_video_to_audio(input_file, audio_path)

        if result is None:
            logger.log("VIDEO_PROCESSING", "ERROR", "Video conversion failed")
            return {"error": "Video conversion failed"}
        logger.log("VIDEO_PROCESSING", "SUCCESS", "Video converted to audio", output_file=str(audio_path))

    elif input_type in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
        logger.log("AUDIO_PROCESSING", "INFO", f"Processing audio file: '{input_file.name}'")
        if input_type != ".wav" or force_wav:
            output_dir = Path("./data/audio")
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = output_dir / f"{input_file.stem}_standardized.wav"
            result = convert_audio_format(input_file, audio_path)
            if result is None:
                logger.log("AUDIO_PROCESSING", "WARNING", "Audio format conversion failed. Using original file.")
                audio_path = input_file
            else:
                logger.log("AUDIO_PROCESSING", "SUCCESS", "Audio format standardized", output_file=str(audio_path))
        else:
            audio_path = input_file
            logger.log("AUDIO_PROCESSING", "INFO", f"Using WAV file directly: {audio_path.name}")

    if denoise or aggressive_denoise:
        logger.log("AUDIO_ENHANCEMENT", "INFO", "Starting audio enhancement", aggressive_mode=aggressive_denoise)
        enhanced_audio_path = enhance_audio(audio_path, aggressive_mode=aggressive_denoise)
        if enhanced_audio_path:
            audio_path = enhanced_audio_path
            logger.log("AUDIO_ENHANCEMENT", "SUCCESS", "Audio enhancement completed")
        else:
            logger.log("AUDIO_ENHANCEMENT", "ERROR", "Audio enhancement failed. Proceeding with original audio.")
    else:
        logger.log("AUDIO_ENHANCEMENT", "INFO", "Audio enhancement skipped by user choice")

    logger.log("MODEL_INIT", "INFO", "Initializing AI models")
    try:
        transcriber = Transcriber(model_name=transcriber_model, senopati_model=senopati_model)
        summarizer = Summarizer(senopati_model)
        logger.log("MODEL_INIT", "SUCCESS", "AI models initialized successfully")
    except Exception as e:
        logger.log("MODEL_INIT", "ERROR", f"Initialization error: {e}")
        return {"error": f"Model initialization failed: {str(e)}"}

    # transcribe
    logger.log("TRANSCRIPTION", "INFO", f"Starting transcription of: {audio_path.name}")
    result, detected_language = transcriber.transcribe(audio_path, language=language)
    if result:
        logger.log("TRANSCRIPTION", "SUCCESS", "Transcription completed",
                   detected_language=detected_language,
                   transcript_length_chars=len(result))
        logger.log("CORRECTION", "INFO", "Applying language rule correction")
        transcript = transcriber.language_rules(result, detected_language)
        logger.log("CORRECTION", "SUCCESS", "Language correction applied")
    else:
        logger.log("TRANSCRIPTION", "ERROR", "Transcription failed")
        return {"error": "Transcription failed"}

    # chunking
    logger.log("TEXT_PROCESSING", "INFO", "Starting text processing and summarization")
    chunks = summarizer.chunk_text(transcript, chunk_size)
    logger.log("TEXT_CHUNKING", "SUCCESS", "Text split into chunks",
               num_chunks=len(chunks),
               chunk_size=chunk_size)

    # clustering
    if len(chunks) > 7:
        logger.log("CLUSTERING", "INFO", "Clustering chunks by topic")
        clusters = summarizer.cluster_chunks(chunks)
        logger.log("CLUSTERING", "SUCCESS", "Topic clustering completed",
                   num_clusters=len(clusters))
    else:
        clusters = {0: chunks}
        logger.log("CLUSTERING", "INFO", "Few Chunks - no clustering needed")

    # summarize
    logger.log("SUMMARIZATION", "INFO", "Generating comprehensive summary")
    cluster_summaries, final_summary = summarizer.get_final_summary(clusters, language=detected_language)
    logger.log("SUMMARIZATION", "SUCCESS", "Final summary generated",
               summary_length_chars=len(final_summary))

    logger.log("PIPELINE_COMPLETE", "SUCCESS", "Pipeline completed successfully")
    logger.save()

    # CLEANUP
    temp_folders  = [Path("data/audio/enhanced"), Path("data/temp"), Path("data/audio")]
    for folder in temp_folders:
        try:
            if folder.exists():
                for file in folder.rglob("*"):
                    if file.is_file():
                        file.unlink()
        except Exception as e:
            logger.log("CLEANUP", "WARNING", f"Failed to clean up {folder}: {e}")

    return {
        "summary": final_summary,
        "transcript": transcript,
        "detected_language": detected_language,
        "chunks": chunks,
        "cluster_summaries": cluster_summaries
    }
