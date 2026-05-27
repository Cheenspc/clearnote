import streamlit as st
import os
import tempfile
import math
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import json
from pydub import AudioSegment

os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
import yt_dlp

load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))

def split_audio(file_path, chunk_minutes=10):
    audio = AudioSegment.from_file(file_path)
    chunk_ms = chunk_minutes * 60 * 1000
    total_chunks = math.ceil(len(audio) / chunk_ms)
    chunks = []
    for i in range(total_chunks):
        start = i * chunk_ms
        end = min((i + 1) * chunk_ms, len(audio))
        chunk = audio[start:end]
        chunk_path = f"{file_path}_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks

def transcribe_chunk(file_path, language=None):
    with open(file_path, "rb") as audio_file:
        kwargs = {"model": "whisper-1", "file": audio_file}
        if language:
            kwargs["language"] = language
        result = openai_client.audio.transcriptions.create(**kwargs)
    return result.text

def transcribe(file_path, language=None):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb < 24:
        return transcribe_chunk(file_path, language)
    else:
        chunks = split_audio(file_path)
        full_transcript = ""
        for i, chunk_path in enumerate(chunks):
            st.info(f"Transcribing chunk {i+1} of {len(chunks)}...")
            full_transcript += transcribe_chunk(chunk_path, language) + " "
            os.unlink(chunk_path)
        return full_transcript.strip()

def split_text_into_chunks(text, max_words=1500):
    """Split text into chunks of roughly max_words words, breaking at sentence boundaries."""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append('. '.join(current_chunk))

    return chunks

def format_transcript(transcript):
    """Clean up transcript in chunks to handle long sermons."""
    words = transcript.split()
    
    # If short enough, process in one shot
    if len(words) <= 1500:
        return format_transcript_chunk(transcript)
    
    # Otherwise chunk it
    text_chunks = split_text_into_chunks(transcript, max_words=1500)
    cleaned_parts = []
    
    for i, chunk in enumerate(text_chunks):
        st.info(f"Cleaning transcript part {i+1} of {len(text_chunks)}...")
        cleaned_parts.append(format_transcript_chunk(chunk))
    
    return "\n\n".join(cleaned_parts)

def format_transcript_chunk(text):
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""You are a transcript editor. Clean up this raw transcript into well-organised, readable paragraphs.

Rules:
- Keep ALL the original words and content — do not summarise or cut anything
- Break into natural paragraphs based on topic shifts, not just full stops
- Fix run-on sentences caused by speech-to-text errors
- Remove filler words like "um", "uh", "you know", "like" only when excessive
- Fix obvious transcription errors (wrong words that don't make sense in context)
- Do NOT add headers or bullet points — just clean flowing paragraphs
- Preserve speaker's natural voice and style

Return only the cleaned transcript, nothing else.

Raw transcript:
{text}"""
        }]
    )
    return message.content[0].text

def download_audio_from_url(url):
    tmp_dir = tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "concurrent_fragment_downloads": 5,
        "quiet": True,
        "no_warnings": True,
        "cookiesfrombrowser": ("chrome",),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    for f in os.listdir(tmp_dir):
        if f.endswith(".mp3"):
            return os.path.join(tmp_dir, f)
    raise Exception("Download completed but MP3 file not found.")

def get_prompt(audio_type, transcript, speaker, date):
    speaker_val = speaker if speaker else "Unknown"
    date_val = date if date else "Unknown"

    metadata_instruction = f"""IMPORTANT: The following values were provided by the user. Copy them exactly into the JSON:
- speaker: "{speaker_val}"
- date: "{date_val}"

"""

    if audio_type == "Sermon":
        return f"""You are a JSON API. You only respond with valid JSON, no markdown, no backticks, no explanation.

{metadata_instruction}Given this sermon transcript, respond with ONLY this JSON object. Be comprehensive and detailed:
{{
    "title": "sermon title or main theme",
    "speaker": "{speaker_val}",
    "date": "{date_val}",
    "audio_type": "Sermon",
    "summary": "4-5 sentence overview of the entire sermon",
    "scriptures": [
        "Book Chapter:Verse — brief note on how it was used",
        "Book Chapter:Verse — brief note on how it was used"
    ],
    "sermon_structure": [
        "Introduction — what the preacher opened with and how he set up the message",
        "Point 1 — describe the first main point in 2 sentences",
        "Point 2 — describe the second main point in 2 sentences",
        "Point 3 — describe the third main point in 2 sentences",
        "Conclusion — how the sermon was closed and what the call to action was"
    ],
    "argument_development": [
        "Step 1 — how the theological argument began",
        "Step 2 — how the argument developed",
        "Step 3 — how the argument reached its conclusion"
    ],
    "key_points": [
        "Key point 1 — explain in 2 sentences what this point is and why it matters",
        "Key point 2 — explain in 2 sentences what this point is and why it matters",
        "Key point 3 — explain in 2 sentences what this point is and why it matters"
    ],
    "quotes": [
        "Most memorable or impactful quote from the sermon, verbatim or close to it",
        "Second most impactful quote"
    ],
    "reformers_thought": {{
        "reformer": "Full name of the most relevant Reformer or Puritan theologian (e.g. John Calvin, Martin Luther, John Owen, Charles Spurgeon, Jonathan Edwards, Herman Bavinck)",
        "relevance": "2-3 sentences explaining why this particular reformer is the most relevant to this sermon's themes",
        "work": "Title of the specific work being referenced",
        "quote": "A specific quote or close paraphrase from that reformer's writings that connects to this sermon's core theme",
        "analysis": "4-6 sentences of in-depth analysis: how would this reformer engage with this sermon? What would they affirm? What would they emphasise differently? How does their theological tradition illuminate or challenge the sermon's argument?"
    }}
}}

Transcript:
{transcript}"""

    elif audio_type == "Meeting":
        return f"""You are a JSON API. You only respond with valid JSON, no markdown, no backticks, no explanation.

{metadata_instruction}Given this meeting transcript, respond with ONLY this JSON object. Be comprehensive and detailed:
{{
    "title": "meeting topic or project name",
    "speaker": "{speaker_val}",
    "date": "{date_val}",
    "audio_type": "Meeting",
    "summary": "3-4 sentence overview of what the meeting was about and what was achieved",
    "decisions": [
        "Decision 1 — what was decided and why, in 2 sentences",
        "Decision 2 — what was decided and why, in 2 sentences"
    ],
    "action_items": [
        "Action 1 — who is responsible and what needs to be done",
        "Action 2 — who is responsible and what needs to be done"
    ],
    "key_points": [
        "Point 1 — main discussion point in 2 sentences",
        "Point 2 — main discussion point in 2 sentences"
    ],
    "blockers": [
        "Any blockers, risks, or unresolved issues mentioned"
    ],
    "quotes": [
        "most important thing said in the meeting"
    ]
}}

Transcript:
{transcript}"""

    elif audio_type == "Other":
        return f"""You are a JSON API. You only respond with valid JSON, no markdown, no backticks, no explanation.

{metadata_instruction}Given this transcript, respond with ONLY this JSON object. Be comprehensive and detailed:
{{
    "title": "main topic or title of the content",
    "speaker": "{speaker_val}",
    "date": "{date_val}",
    "audio_type": "Other",
    "summary": "4-5 sentence overview of the content",
    "key_points": [
        "Key point 1 — explain in 2 sentences what this point is and why it matters",
        "Key point 2 — explain in 2 sentences what this point is and why it matters",
        "Key point 3 — explain in 2 sentences what this point is and why it matters",
        "Key point 4 — explain in 2 sentences what this point is and why it matters"
    ],
    "quotes": [
        "Most memorable or impactful quote from the content",
        "Second most impactful quote"
    ]
}}

Transcript:
{transcript}"""

def summarize(transcript, speaker, date, audio_type):
    prompt = get_prompt(audio_type, transcript, speaker, date)
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text
    start = raw.find('{')
    end = raw.rfind('}') + 1
    parsed = json.loads(raw[start:end])
    if speaker and parsed.get('speaker') in ('Unknown', '', None):
        parsed['speaker'] = speaker
    if date and parsed.get('date') in ('Unknown', '', None):
        parsed['date'] = date
    return parsed

def translate_notes(data, target_language):
    lang_instruction = {
        "Simplified Chinese": "Translate all text values in this JSON into Simplified Chinese (简体中文). Keep keys in English.",
        "Traditional Chinese": "Translate all text values in this JSON into Traditional Chinese (繁體中文). Keep keys in English.",
    }
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""You are a JSON API. You only respond with valid JSON, no markdown, no backticks, no explanation.

{lang_instruction[target_language]}

JSON to translate:
{json.dumps(data, ensure_ascii=False)}"""
        }]
    )
    raw = message.content[0].text
    start = raw.find('{')
    end = raw.rfind('}') + 1
    return json.loads(raw[start:end])

def format_notes_as_text(data, audio_type):
    lines = []
    lines.append(f"CLEARNOTE — {data.get('audio_type', audio_type).upper()}")
    lines.append("=" * 60)
    lines.append(f"Title:   {data.get('title', '')}")
    lines.append(f"Speaker: {data.get('speaker', 'Unknown')}")
    lines.append(f"Date:    {data.get('date', 'Unknown')}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(data.get('summary', ''))
    lines.append("")

    if audio_type == "Sermon":
        lines.append("SCRIPTURES")
        lines.append("-" * 40)
        for s in data.get('scriptures', []):
            lines.append(f"• {s}")
        lines.append("")
        lines.append("SERMON STRUCTURE")
        lines.append("-" * 40)
        for i, s in enumerate(data.get('sermon_structure', []), 1):
            lines.append(f"{i}. {s}")
        lines.append("")
        lines.append("ARGUMENT DEVELOPMENT")
        lines.append("-" * 40)
        for i, s in enumerate(data.get('argument_development', []), 1):
            lines.append(f"{i}. {s}")
        lines.append("")
        lines.append("KEY POINTS")
        lines.append("-" * 40)
        for p in data.get('key_points', []):
            lines.append(f"• {p}")
        lines.append("")
        lines.append("NOTABLE QUOTES")
        lines.append("-" * 40)
        for q in data.get('quotes', []):
            lines.append(f'"{q}"')
        lines.append("")
        rt = data.get('reformers_thought', {})
        if rt:
            lines.append("REFORMER'S THOUGHT")
            lines.append("-" * 40)
            lines.append(f"Reformer: {rt.get('reformer', '')}")
            lines.append(f"Work:     {rt.get('work', '')}")
            lines.append("")
            lines.append(f"Why this reformer: {rt.get('relevance', '')}")
            lines.append("")
            lines.append(f'Quote: "{rt.get("quote", "")}"')
            lines.append("")
            lines.append(f"Analysis: {rt.get('analysis', '')}")

    elif audio_type == "Meeting":
        lines.append("DECISIONS")
        lines.append("-" * 40)
        for d in data.get('decisions', []):
            lines.append(f"• {d}")
        lines.append("")
        lines.append("ACTION ITEMS")
        lines.append("-" * 40)
        for a in data.get('action_items', []):
            lines.append(f"• {a}")
        lines.append("")
        lines.append("KEY POINTS")
        lines.append("-" * 40)
        for p in data.get('key_points', []):
            lines.append(f"• {p}")
        lines.append("")
        lines.append("BLOCKERS")
        lines.append("-" * 40)
        for b in data.get('blockers', []):
            lines.append(f"• {b}")
        lines.append("")
        lines.append("QUOTES")
        lines.append("-" * 40)
        for q in data.get('quotes', []):
            lines.append(f'"{q}"')

    elif audio_type == "Other":
        lines.append("KEY POINTS")
        lines.append("-" * 40)
        for p in data.get('key_points', []):
            lines.append(f"• {p}")
        lines.append("")
        lines.append("QUOTES")
        lines.append("-" * 40)
        for q in data.get('quotes', []):
            lines.append(f'"{q}"')

    return "\n".join(lines)

def render_notes(data, audio_type):
    st.markdown(f"**Type:** {data.get('audio_type', audio_type)}")
    st.markdown(f"**Speaker:** {data.get('speaker', 'Unknown')}")
    st.markdown(f"**Date:** {data.get('date', 'Unknown')}")
    st.markdown(f"**Title:** {data.get('title', '')}")
    st.markdown(f"_{data.get('summary', '')}_")

    if audio_type == "Sermon":
        st.markdown("**📖 Scriptures**")
        for s in data.get('scriptures', []):
            st.markdown(f"• {s}")
        st.markdown("**🏗️ Sermon Structure**")
        for i, step in enumerate(data.get('sermon_structure', []), 1):
            st.markdown(f"**{i}.** {step}")
        st.markdown("**🧠 Argument Development**")
        for i, step in enumerate(data.get('argument_development', []), 1):
            st.markdown(f"**{i}.** {step}")
        st.markdown("**🎯 Key Points**")
        for p in data.get('key_points', []):
            st.markdown(f"• {p}")
        st.markdown("**📌 Notable Quotes**")
        for q in data.get('quotes', []):
            st.markdown(f'• "{q}"')
        rt = data.get('reformers_thought', {})
        if rt:
            st.markdown("---")
            st.markdown("**⛪ Reformer's Thought**")
            st.markdown(f"**{rt.get('reformer', '')}** — *{rt.get('work', '')}*")
            st.markdown(f"**Why this reformer:** {rt.get('relevance', '')}")
            st.markdown(f"> {rt.get('quote', '')}")
            st.markdown(f"**Analysis:** {rt.get('analysis', '')}")

    elif audio_type == "Meeting":
        st.markdown("**✅ Decisions**")
        for d in data.get('decisions', []):
            st.markdown(f"• {d}")
        st.markdown("**📌 Action Items**")
        for a in data.get('action_items', []):
            st.markdown(f"• {a}")
        st.markdown("**🎯 Key Points**")
        for p in data.get('key_points', []):
            st.markdown(f"• {p}")
        st.markdown("**⚠️ Blockers**")
        for b in data.get('blockers', []):
            st.markdown(f"• {b}")
        st.markdown("**💬 Quotes**")
        for q in data.get('quotes', []):
            st.markdown(f'• "{q}"')

    elif audio_type == "Other":
        st.markdown("**🎯 Key Points**")
        for p in data.get('key_points', []):
            st.markdown(f"• {p}")
        st.markdown("**📌 Quotes**")
        for q in data.get('quotes', []):
            st.markdown(f'• "{q}"')

# --- STREAMLIT UI ---
st.set_page_config(page_title="ClearNote", page_icon="🎧", layout="wide")

# Password gate
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("Enter password", type="password")
    if pwd == get_secret("APP_PASSWORD"):
        st.session_state.authenticated = True
        st.rerun()
    elif pwd:
        st.error("Wrong password")
    st.stop()

st.title("🎧 ClearNote")
st.markdown("Upload an audio recording and get structured notes and transcript.")

col1, col2 = st.columns(2)
with col1:
    speaker = st.text_input("Speaker / Host name", placeholder="e.g. Pastor David, Dr. Ahmad")
with col2:
    date = st.text_input("Date", placeholder="e.g. 25 May 2026")

audio_type = st.selectbox(
    "Audio type",
    ["Sermon", "Meeting", "Other"],
    help="Choose the type of audio — each type produces a different output format"
)

force_english = st.checkbox("🔤 Force English transcription", value=True, help="Enable this if the speaker speaks English but Whisper transcribes in the wrong language. Disable for non-English audio.")

output_language = st.selectbox(
    "Output language",
    ["English only", "Simplified Chinese (简体中文)", "Traditional Chinese (繁體中文)", "English + Simplified Chinese", "English + Traditional Chinese"],
    help="Notes will be translated. Transcript always stays in the original spoken language."
)

st.markdown("### Audio source")
input_method = st.radio("Choose input method", ["Upload a file", "Paste a video URL"], horizontal=True)

uploaded_files = None
video_url = None

if input_method == "Upload a file":
    uploaded_files = st.file_uploader(
        "Upload audio file(s)",
        type=["mp3", "mp4", "m4a", "wav"],
        accept_multiple_files=True,
        help="Upload multiple files if your recording was split into parts — they will be stitched in order."
    )
else:
    video_url = st.text_input(
        "Paste video URL",
        placeholder="YouTube, Vimeo, or any public video URL"
    )
    st.info("💡 Make sure you are logged into a **non-Premium** YouTube account in Chrome for best results.")

ready = (uploaded_files and len(uploaded_files) > 0) or (video_url and video_url.strip() != "")

if ready and st.button("Generate Notes"):
    st.session_state.pop('results', None)
    tmp_path = None

    if input_method == "Upload a file" and uploaded_files:
        if len(uploaded_files) == 1:
            ext = os.path.splitext(uploaded_files[0].name)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_files[0].read())
                tmp_path = tmp.name
            file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            if file_size_mb >= 24:
                st.info(f"Large file detected ({file_size_mb:.1f}MB) — splitting into chunks automatically...")
        else:
            st.info(f"{len(uploaded_files)} files detected — transcribing and stitching in order...")
    else:
        with st.spinner("Downloading audio from URL..."):
            try:
                tmp_path = download_audio_from_url(video_url.strip())
                st.success("Audio downloaded successfully!")
            except Exception as e:
                st.error(f"Download failed: {e}\n\nTip: Make sure the video is public and you are logged into a non-Premium YouTube account in Chrome.")
                st.stop()

    if input_method == "Upload a file" and uploaded_files and len(uploaded_files) > 1:
        whisper_lang = "en" if force_english else None
        all_transcripts = []
        for i, uf in enumerate(uploaded_files):
            ext = os.path.splitext(uf.name)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uf.read())
                part_path = tmp.name
            with st.spinner(f"Transcribing part {i+1} of {len(uploaded_files)}..."):
                all_transcripts.append(transcribe(part_path, whisper_lang))
            os.unlink(part_path)
        raw_transcript = " ".join(all_transcripts)
    else:
        whisper_lang = "en" if force_english else None
        with st.spinner("Transcribing audio..."):
            raw_transcript = transcribe(tmp_path, whisper_lang)

    with st.spinner("Cleaning up transcript..."):
        transcript = format_transcript(raw_transcript)

    with st.spinner(f"Generating {audio_type} notes..."):
        english_data = summarize(transcript, speaker, date, audio_type)

    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    chinese_data = None
    if output_language in ["Simplified Chinese (简体中文)", "English + Simplified Chinese"]:
        with st.spinner("Translating to Simplified Chinese..."):
            chinese_data = translate_notes(english_data, "Simplified Chinese")
    elif output_language in ["Traditional Chinese (繁體中文)", "English + Traditional Chinese"]:
        with st.spinner("Translating to Traditional Chinese..."):
            chinese_data = translate_notes(english_data, "Traditional Chinese")

    st.session_state['results'] = {
        'english_data': english_data,
        'chinese_data': chinese_data,
        'transcript': transcript,
        'audio_type': audio_type,
        'output_language': output_language,
    }

if 'results' in st.session_state:
    r = st.session_state['results']
    english_data = r['english_data']
    chinese_data = r['chinese_data']
    transcript = r['transcript']
    audio_type = r['audio_type']
    output_language = r['output_language']

    st.success("Done!")

    if output_language == "English only":
        st.subheader("📋 Notes")
        render_notes(english_data, audio_type)
        st.download_button("📥 Download Notes", format_notes_as_text(english_data, audio_type), file_name="clearnote-notes.txt", mime="text/plain")

    elif output_language in ["Simplified Chinese (简体中文)", "Traditional Chinese (繁體中文)"]:
        lang_label = "简体中文" if "Simplified" in output_language else "繁體中文"
        st.subheader(f"📋 Notes ({lang_label})")
        render_notes(chinese_data, audio_type)
        st.download_button(f"📥 Download Notes ({lang_label})", format_notes_as_text(chinese_data, audio_type), file_name="clearnote-notes-zh.txt", mime="text/plain")

    elif output_language in ["English + Simplified Chinese", "English + Traditional Chinese"]:
        lang_label = "简体中文" if "Simplified" in output_language else "繁體中文"
        tab_en, tab_zh = st.tabs(["English", lang_label])
        with tab_en:
            render_notes(english_data, audio_type)
            st.download_button("📥 Download Notes (EN)", format_notes_as_text(english_data, audio_type), file_name="clearnote-notes-en.txt", mime="text/plain")
        with tab_zh:
            render_notes(chinese_data, audio_type)
            st.download_button(f"📥 Download Notes ({lang_label})", format_notes_as_text(chinese_data, audio_type), file_name="clearnote-notes-zh.txt", mime="text/plain")

    st.subheader("📝 Full Transcript")
    st.markdown(transcript)
    st.download_button("📥 Download Transcript", transcript, file_name="clearnote-transcript.txt", mime="text/plain")
