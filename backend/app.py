import os
from flask import Flask, jsonify, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from youtube_transcript_api import YouTubeTranscriptApi
from pymongo import MongoClient

app = Flask(__name__)

# Load Google Cloud credentials from environment variable
google_credentials_json = os.getenv('GOOGLE_CLOUD_CREDENTIALS_JSON')

if google_credentials_json:
    with open('/tmp/google-credentials.json', 'w') as f:
        f.write(google_credentials_json)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/google-credentials.json'
else:
    raise Exception("Google Cloud credentials not found in environment variables.")

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['youtube_summarizer']
summaries_collection = db['summaries']

# Initialize summarizer as None to delay model loading
summarizer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    global summarizer
    try:
        data = request.json
        video_url = data.get('video_url')

        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400

        # Safe extraction of YouTube video ID
        video_id = video_url.split('v=')[-1] if 'v=' in video_url else None
        if not video_id:
            return jsonify({'error': 'Invalid video URL provided'}), 400

        # Check if the summary already exists in the database
        existing_summary = summaries_collection.find_one({"video_id": video_id})
        if existing_summary:
            return jsonify({'summary': existing_summary['summary'], 'analysis': ''})

        # Fetch transcript
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            return jsonify({'error': 'Could not retrieve transcript. The video might not have subtitles or is restricted.'}), 500

        # Ensure transcript is not empty
        if not transcript:
            return jsonify({'error': 'Transcript is empty or unavailable for this video'}), 500

        # Combine transcript text
        transcript_text = " ".join([item['text'] for item in transcript])

        # Ensure the transcript text is not too short
        if len(transcript_text.split()) < 50:
            return jsonify({'error': 'Transcript too short to summarize effectively'}), 400

        # Lazy load the model only when needed
        if summarizer is None:
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

        # Truncate the text if it's too long
        max_input_length = 1024
        tokens = tokenizer.encode(transcript_text, truncation=True, max_length=max_input_length)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

        summary = summarizer(truncated_text, max_length=250, min_length=100, do_sample=False, clean_up_tokenization_spaces=True)[0]['summary_text']

        # Store the summary in MongoDB
        summaries_collection.insert_one({"video_id": video_id, "summary": summary})

        return jsonify({'summary': summary, 'analysis': ''})

    except IndexError as e:
        return jsonify({'error': f'An index error occurred: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port)
