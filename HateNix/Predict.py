import os
import pickle
import csv
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from PIL import Image, ImageTk
import requests
from io import BytesIO
from threading import Thread
from queue import Queue
from transformers import pipeline
from model import HateSpeechModel

class YouTubeCommentModerator:
    def __init__(self, root):
        self.root = root
        self.root.title("HateNIx")
        self.root.geometry("1200x800")

        # YouTube API configuration
        self.client_secrets_file = #add your client_secret files path here
        print(f"Client secrets file set to: {self.client_secrets_file}")
        self.credentials = None
        self.youtube = None

        # Initialize hate speech classifiers
        try:
            self.roberta_classifier = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target",
                device=-1  # Use CPU (change to 'mps' for Apple Silicon if available)
            )
            print("RoBERTa model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load RoBERTa model: {e}. Ensure PyTorch is installed.")
            print(f"RoBERTa load error: {e}")
            self.roberta_classifier = None

        # Initialize custom hate speech model
        self.custom_classifier = None
        self.model_dir = '/Users/vansh./PycharmProjects/model/'
        self.load_custom_model()

        # Data variables
        self.channel_id = ""
        self.videos_data = []
        self.comments_data = []
        self.current_video = None
        self.filter_mode = "all"
        self.thumbnail_queue = Queue()
        self.thumbnail_cache = {}

        # UI Setup
        self.setup_ui()

        # Start thumbnail loader thread
        self.thumbnail_thread = Thread(target=self.thumbnail_loader, daemon=True)
        self.thumbnail_thread.start()

    def load_custom_model(self):
        """Load the custom trained hate speech model"""
        try:
            model_path = os.path.join(self.model_dir, 'hate_speech_model.pkl')
            dataset_path = os.path.join(self.model_dir, 'hindi-hinglish-hate-speech-dataset.txt')
            self.custom_classifier = HateSpeechModel(model_path=model_path, dataset_path=dataset_path)
            print(f"Custom model loaded from {model_path}")
        except FileNotFoundError as e:
            messagebox.showwarning("Warning", f"Custom model not found: {e}. Using RoBERTa only.")
            print(f"Custom model error: {e}")
            self.custom_classifier = None
        except Exception as e:
            messagebox.showwarning("Warning", f"Custom model error: {e}. Using RoBERTa only.")
            print(f"Custom model error: {e}")
            self.custom_classifier = None

    def setup_ui(self):
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Hateful.Treeview', background='#ffdddd', foreground='black')
        style.configure('Neutral.Treeview', background='#ffffff', foreground='black')
        style.configure('Video.TFrame', background='#ffffff', borderwidth=1, relief='solid')
        style.configure('VideoSelected.TFrame', background='#e0e0ff', borderwidth=2, relief='solid')

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.login_frame = ttk.Frame(self.main_frame)
        self.setup_login_frame()
        self.app_frame = ttk.Frame(self.main_frame)

    def setup_login_frame(self):
        for widget in self.login_frame.winfo_children():
            widget.destroy()

        self.login_frame.pack(fill=tk.BOTH, expand=True)

        try:
            img = Image.open('youtube_logo.png')
            img = img.resize((200, 140), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            logo_label = ttk.Label(self.login_frame, image=self.logo)
            logo_label.pack(pady=20)
        except FileNotFoundError:
            logo_label = ttk.Label(self.login_frame, text="YouTube Comment Moderator", style='Header.TLabel')
            logo_label.pack(pady=20)

        header = ttk.Label(self.login_frame, text="HateNIx", style='Header.TLabel')
        header.pack(pady=10)

        instructions = ttk.Label(self.login_frame,
                                text="Login with your YouTube account to moderate comments",
                                wraplength=400)
        instructions.pack(pady=10)

        login_btn = ttk.Button(self.login_frame, text="Login with YouTube",
                              command=self.authenticate_oauth)
        login_btn.pack(pady=20, ipadx=20, ipady=5)
        print("Login frame setup complete")

    def setup_app_frame(self):
        for widget in self.app_frame.winfo_children():
            widget.destroy()

        self.login_frame.pack_forget()
        self.app_frame.pack(fill=tk.BOTH, expand=True)
        print("Setting up app frame")

        header_frame = ttk.Frame(self.app_frame)
        header_frame.pack(fill=tk.X, pady=10)

        welcome_label = ttk.Label(header_frame,
                                 text=f"Managing comments for channel: {self.channel_id}",
                                 style='Header.TLabel')
        welcome_label.pack(side=tk.LEFT, padx=10)

        logout_btn = ttk.Button(header_frame, text="Logout", command=self.logout)
        logout_btn.pack(side=tk.RIGHT, padx=10)

        content_frame = ttk.Frame(self.app_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        self.video_panel = ttk.Frame(content_frame, width=400)
        self.video_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.video_panel.pack_propagate(False)

        video_header = ttk.Frame(self.video_panel)
        video_header.pack(fill=tk.X, pady=5)

        ttk.Label(video_header, text="Your Videos", style='Header.TLabel').pack(side=tk.LEFT)

        refresh_btn = ttk.Button(video_header, text="↻", width=3,
                                command=self.fetch_channel_videos)
        refresh_btn.pack(side=tk.RIGHT)

        self.video_canvas = tk.Canvas(self.video_panel, bg='#f0f0f0', highlightthickness=0)
        self.video_scrollbar = ttk.Scrollbar(self.video_panel, orient=tk.VERTICAL,
                                            command=self.video_canvas.yview)
        self.video_canvas.configure(yscrollcommand=self.video_scrollbar.set)

        self.video_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_inner_frame = ttk.Frame(self.video_canvas)
        self.video_canvas.create_window((0, 0), window=self.video_inner_frame, anchor=tk.NW)

        self.video_inner_frame.bind("<Configure>",
                                   lambda e: self.video_canvas.configure(
                                       scrollregion=self.video_canvas.bbox("all")))

        comments_panel = ttk.Frame(content_frame)
        comments_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        filter_frame = ttk.Frame(comments_panel)
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=5)

        self.filter_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text="All Comments", variable=self.filter_var,
                        value="all", command=self.apply_filters).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(filter_frame, text="Potentially Harmful", variable=self.filter_var,
                        value="hateful", command=self.apply_filters).pack(side=tk.LEFT, padx=5)

        comments_display_frame = ttk.Frame(comments_panel)
        comments_display_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(comments_display_frame,
                                 columns=('id', 'author', 'comment', 'date', 'sentiment'),
                                 show='headings', height=20)
        self.tree.heading('id', text='Comment ID')
        self.tree.heading('author', text='Author')
        self.tree.heading('comment', text='Comment')
        self.tree.heading('date', text='Date')
        self.tree.heading('sentiment', text='Sentiment')

        self.tree.column('id', width=150, anchor=tk.W, stretch=True, minwidth=100)
        self.tree.column('author', width=200, anchor=tk.W, stretch=True, minwidth=150)
        self.tree.column('comment', width=400, anchor=tk.W, stretch=True, minwidth=300)
        self.tree.column('date', width=150, anchor=tk.W, stretch=True, minwidth=100)
        self.tree.column('sentiment', width=120, anchor=tk.W, stretch=True, minwidth=100)

        scrollbar = ttk.Scrollbar(comments_display_frame, orient=tk.VERTICAL,
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.debug_label = ttk.Label(comments_display_frame, text="Debug: Loading...")
        self.debug_label.pack(pady=5)

        button_frame = ttk.Frame(comments_panel)
        button_frame.pack(fill=tk.X, pady=5)

        delete_selected_btn = ttk.Button(button_frame, text="Delete Selected",
                                        command=self.delete_selected_comments)
        delete_selected_btn.pack(side=tk.LEFT, padx=5)
        print("Delete Selected button added to UI")

        delete_hateful_btn = ttk.Button(button_frame, text="Delete Hateful Comments",
                                       command=self.delete_hateful_comments)
        delete_hateful_btn.pack(side=tk.LEFT, padx=5)
        print("Delete Hateful Comments button added to UI")

        export_btn = ttk.Button(button_frame, text="Export Comments",
                                command=self.export_comments_to_csv)
        export_btn.pack(side=tk.LEFT, padx=5)

        analyze_btn = ttk.Button(button_frame, text="Re-analyze Comments",
                                 command=self.analyze_comments)
        analyze_btn.pack(side=tk.LEFT, padx=5)

        file_analyze_btn = ttk.Button(button_frame, text="Analyze Comments from File",
                                      command=self.analyze_comments_from_file)
        file_analyze_btn.pack(side=tk.LEFT, padx=5)
        print("App frame setup complete")

    def thumbnail_loader(self):
        while True:
            video_id, url = self.thumbnail_queue.get()
            try:
                if video_id in self.thumbnail_cache:
                    continue
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content))
                img = img.resize((320, 180), Image.LANCZOS)
                self.thumbnail_cache[video_id] = ImageTk.PhotoImage(img)
                self.root.after(0, self.update_video_thumbnail, video_id)
            except Exception as e:
                print(f"Error loading thumbnail {video_id}: {e}")
            finally:
                self.thumbnail_queue.task_done()

    def update_video_thumbnail(self, video_id):
        for widget in self.video_inner_frame.winfo_children():
            if hasattr(widget, 'video_id') and widget.video_id == video_id:
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label) and hasattr(child, 'is_thumbnail'):
                        if video_id in self.thumbnail_cache:
                            child.configure(image=self.thumbnail_cache[video_id])
                        break
                break

    def create_video_card(self, video_data):
        frame = ttk.Frame(self.video_inner_frame, style='Video.TFrame')
        frame.pack(fill=tk.X, pady=5, padx=5)
        frame.video_id = video_data['id']
        thumbnail_label = ttk.Label(frame)
        thumbnail_label.pack(fill=tk.X)
        thumbnail_label.is_thumbnail = True
        self.thumbnail_queue.put((video_data['id'], video_data['thumbnail']))
        title_label = ttk.Label(frame, text=video_data['title'], wraplength=310)
        title_label.pack(fill=tk.X, padx=5, pady=5)
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(info_frame, text=video_data['date']).pack(side=tk.LEFT)
        ttk.Label(info_frame, text=f"👁 {video_data['viewCount']}").pack(side=tk.RIGHT)
        frame.bind("<Button-1>", lambda e, vid=video_data['id']: self.select_video(vid))
        for child in frame.winfo_children():
            child.bind("<Button-1>", lambda e, vid=video_data['id']: self.select_video(vid))
        return frame

    def select_video(self, video_id):
        print(f"Selected video ID: {video_id}")
        self.current_video = video_id
        for widget in self.video_inner_frame.winfo_children():
            if hasattr(widget, 'video_id'):
                widget.configure(style='VideoSelected.TFrame' if widget.video_id == video_id else 'Video.TFrame')
        self.load_video_comments()

    def authenticate_oauth(self):
        try:
            print(f"Starting authentication with client_secrets_file: {self.client_secrets_file}")
            if os.path.exists('token.pickle'):
                os.remove('token.pickle')
            if not os.path.exists(self.client_secrets_file):
                messagebox.showerror("Error", f"Client secrets file not found at {self.client_secrets_file}")
                print(f"File check failed: {os.listdir('/Users/vansh./PycharmProjects/model/')}")
                return
            flow = InstalledAppFlow.from_client_secrets_file(
                self.client_secrets_file,
                scopes=['https://www.googleapis.com/auth/youtube.force-ssl'],
                redirect_uri='http://localhost:8080'
            )
            self.credentials = flow.run_local_server(
                port=8080,
                prompt='consent',
                authorization_prompt_message='Please authorize access to your YouTube account',
                open_browser=True
            )
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.credentials, token)
            self.youtube = build('youtube', 'v3', credentials=self.credentials)
            self.get_channel_id()
        except Exception as e:
            messagebox.showerror("Error", f"Authentication failed: {str(e)}")
            print(f"Authentication error: {e}")
            if os.path.exists('token.pickle'):
                os.remove('token.pickle')
            self.setup_login_frame()

    def get_channel_id(self):
        try:
            request = self.youtube.channels().list(
                part="id,snippet,statistics",
                mine=True
            )
            response = request.execute()
            if 'items' in response and response['items']:
                self.channel_id = response['items'][0]['id']
                self.setup_app_frame()
                self.fetch_channel_videos()
            else:
                messagebox.showerror("Error", "Could not detect channel ID")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get channel ID: {str(e)}")

    def fetch_channel_videos(self):
        if not self.youtube:
            messagebox.showerror("Error", "Not authenticated with YouTube")
            return
        try:
            self.videos_data = []
            next_page_token = None
            for widget in self.video_inner_frame.winfo_children():
                widget.destroy()
            channels_response = self.youtube.channels().list(
                part="contentDetails",
                id=self.channel_id
            ).execute()
            if not channels_response.get('items'):
                messagebox.showerror("Error", "No channel data found")
                return
            uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            while True:
                playlist_request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                playlist_response = playlist_request.execute()
                video_ids = [item['contentDetails']['videoId'] for item in playlist_response.get('items', [])]
                if video_ids:
                    videos_request = self.youtube.videos().list(
                        part="snippet,statistics",
                        id=",".join(video_ids)
                    )
                    videos_response = videos_request.execute()
                    for video in videos_response.get('items', []):
                        thumbnails = video['snippet']['thumbnails']
                        thumbnail_url = thumbnails.get('high', {}).get('url',
                                                                      thumbnails.get('medium', {}).get('url',
                                                                                                      thumbnails.get(
                                                                                                          'default',
                                                                                                          {}).get(
                                                                                                          'url', '')))
                        self.videos_data.append({
                            'id': video['id'],
                            'title': video['snippet']['title'],
                            'thumbnail': thumbnail_url,
                            'date': video['snippet']['publishedAt'][:10],
                            'viewCount': video['statistics'].get('viewCount', 'N/A')
                        })
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token or len(self.videos_data) >= 100:
                    break
            if not self.videos_data:
                messagebox.showinfo("Info", "No videos found for this channel")
                return
            for video in self.videos_data:
                self.create_video_card(video)
            self.video_canvas.update_idletasks()
            self.video_canvas.configure(scrollregion=self.video_canvas.bbox("all"))
            if self.videos_data:
                self.root.after(100, lambda: self.select_video(self.videos_data[0]['id']))
        except HttpError as e:
            messagebox.showerror("Error", f"API error: {str(e)}. Check quota or permissions.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch videos: {str(e)}")

    def load_video_comments(self):
        if not self.current_video:
            return
        try:
            self.comments_data = []
            next_page_token = None
            for item in self.tree.get_children():
                self.tree.delete(item)
            while True:
                response = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=self.current_video,
                    textFormat='plainText',
                    maxResults=100,
                    pageToken=next_page_token
                ).execute()
                if 'items' in response and not response['items']:
                    messagebox.showinfo("Info", "No comments available or comments disabled")
                    break
                if 'items' in response:
                    for comment in response['items']:
                        top_comment = comment['snippet']['topLevelComment']['snippet']
                        comment_id = comment['id']
                        author = top_comment['authorDisplayName']
                        text = top_comment['textDisplay']
                        published_at = top_comment['publishedAt']
                        self.comments_data.append({
                            'id': comment_id,
                            'author': author,
                            'comment': text,
                            'date': published_at,
                            'sentiment': 'pending',
                            'score': 0
                        })
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            print(f"Fetched {len(self.comments_data)} comments")
            self.analyze_comments()
        except HttpError as e:
            messagebox.showerror("Error", f"API error: {e.content.decode()}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")

    def analyze_comments(self):
        if not self.comments_data:
            print("No comments to analyze")
            self.apply_filters()
            return
        if not self.roberta_classifier and not self.custom_classifier:
            messagebox.showerror("Error", "No ML models loaded")
            for comment in self.comments_data:
                comment['sentiment'] = 'unknown'
                comment['score'] = 0
            self.apply_filters()
            return
        try:
            print(f"Analyzing {len(self.comments_data)} comments")
            batch_size = 10
            for i in range(0, len(self.comments_data), batch_size):
                batch = self.comments_data[i:i + batch_size]
                texts = [item['comment'] for item in batch]
                roberta_results = self.roberta_classifier(texts) if self.roberta_classifier else [{'label': 'unknown', 'score': 0} for _ in texts]
                custom_results = self.custom_classifier.predict(texts, threshold=0.7) if self.custom_classifier else [(0, 0.0) for _ in texts]
                print(f"RoBERTa results for batch {i//batch_size}: {roberta_results}")
                print(f"Custom results for batch {i//batch_size}: {custom_results}")
                for j, (comment, roberta, custom) in enumerate(zip(batch, roberta_results, custom_results)):
                    roberta_label = roberta['label'] if self.roberta_classifier else 'unknown'
                    roberta_score = roberta['score'] if self.roberta_classifier else 0.0
                    custom_pred, custom_score = custom
                    final_sentiment = 'hate' if (roberta_label == 'hate' or custom_pred == 1) else 'nothate'
                    final_score = max(roberta_score, custom_score) if final_sentiment == 'hate' else min(roberta_score, custom_score)
                    comment['sentiment'] = final_sentiment
                    comment['score'] = final_score
                    print(f"Comment {comment['id']}: Text='{comment['comment']}', Final Sentiment={final_sentiment}, Score={final_score:.2f}")
            if self.custom_classifier:
                stats = self.custom_classifier.get_detection_stats()
                print(f"Detection stats: {stats}")
            self.apply_filters()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze comments: {str(e)}")
            print(f"Analysis error: {e}")

    def analyze_comments_from_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.comments_data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for i, row in enumerate(reader):
                    if len(row) >= 1:
                        text = row[0].strip()
                        self.comments_data.append({
                            'id': f"file_{i}",
                            'author': 'Unknown',
                            'comment': text,
                            'date': datetime.datetime.now().isoformat(),
                            'sentiment': 'pending',
                            'score': 0
                        })
            if not self.comments_data:
                messagebox.showwarning("Warning", "No valid comments found in file")
                return
            print(f"Loaded {len(self.comments_data)} comments from {file_path}")
            self.analyze_comments()
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file: {str(e)}")

    def apply_filters(self):
        filter_type = self.filter_var.get()
        print(f"Applying filter: {filter_type}")
        for item in self.tree.get_children():
            self.tree.delete(item)
        for comment in self.comments_data:
            if filter_type == "all" or (filter_type == "hateful" and comment['sentiment'] == 'hate'):
                tag = 'hateful' if comment['sentiment'] == 'hate' else 'neutral'
                print(f"Adding comment: {comment['id']}, Sentiment: {comment['sentiment']}")
                item_id = self.tree.insert('', tk.END,
                                          values=(comment['id'], comment['author'],
                                                  self.shorten_comment(comment['comment']),
                                                  comment['date'][:10], comment['sentiment']),
                                          tags=(tag,))
                self.tree.see(item_id)
                self.tree.update()
        self.tree.tag_configure('hateful', background='#ffdddd', foreground='black')
        self.tree.tag_configure('neutral', background='#ffffff', foreground='black')
        self.tree.yview_moveto(0)
        print(f"Visible rows in Treeview: {len(self.tree.get_children())}")
        self.debug_label.config(text=f"Debug: {len(self.tree.get_children())} rows loaded")
        self.root.update_idletasks()

    def delete_hateful_comments(self):
        if not self.comments_data:
            messagebox.showwarning("Warning", "No comments to delete")
            return
        hateful_comments = [comment for comment in self.comments_data if comment['sentiment'] == 'hate']
        if not hateful_comments:
            messagebox.showinfo("Info", "No hateful comments to delete")
            return
        confirm = messagebox.askyesno("Confirm",
                                     f"Are you sure you want to delete {len(hateful_comments)} hateful comments?")
        if not confirm:
            return
        success_count = 0
        for comment in hateful_comments[:]:  # Create a copy to avoid modifying list while iterating
            try:
                if not comment['id'].startswith('file_') and self.youtube:
                    print(f"Deleting comment {comment['id']}")
                    self.youtube.comments().setModerationStatus(
                        id=comment['id'],
                        moderationStatus="rejected"
                    ).execute()
                    self.comments_data.remove(comment)
                    success_count += 1
                elif comment['id'].startswith('file_'):
                    self.comments_data.remove(comment)
                    success_count += 1
            except HttpError as e:
                print(f"Error deleting comment {comment['id']}: {e}")
                messagebox.showwarning("Warning", f"Failed to delete comment {comment['id']}: {e}")
        messagebox.showinfo("Result", f"Successfully deleted {success_count} hateful comments")
        self.apply_filters()

    def delete_selected_comments(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No comments selected")
            return
        confirm = messagebox.askyesno("Confirm",
                                     f"Are you sure you want to delete {len(selected_items)} comments?")
        if not confirm:
            return
        success_count = 0
        for item in selected_items:
            comment_id = self.tree.item(item)['values'][0]
            try:
                if not comment_id.startswith('file_') and self.youtube:
                    print(f"Deleting selected comment {comment_id}")
                    self.youtube.comments().setModerationStatus(
                        id=comment_id,
                        moderationStatus="rejected"
                    ).execute()
                for comment in self.comments_data[:]:
                    if comment['id'] == comment_id:
                        self.comments_data.remove(comment)
                        break
                self.tree.delete(item)
                success_count += 1
            except HttpError as e:
                print(f"Error deleting comment {comment_id}: {e}")
                messagebox.showwarning("Warning", f"Failed to delete comment {comment_id}: {e}")
        messagebox.showinfo("Result", f"Successfully deleted {success_count} comments")
        self.apply_filters()

    def shorten_comment(self, text, max_length=100):
        return text if len(text) <= max_length else text[:max_length - 3] + "..."

    def export_comments_to_csv(self):
        if not self.comments_data:
            messagebox.showwarning("Warning", "No comments to export")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"youtube_comments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not file_path:
            return
        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Comment ID', 'Author', 'Comment', 'Date', 'Sentiment', 'Confidence Score'])
                for comment in self.comments_data:
                    writer.writerow([
                        comment['id'],
                        comment['author'],
                        comment['comment'],
                        comment['date'],
                        comment['sentiment'],
                        comment['score']
                    ])
            messagebox.showinfo("Success", f"Comments exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export comments: {e}")

    def logout(self):
        self.credentials = None
        self.youtube = None
        self.channel_id = ""
        self.videos_data = []
        self.comments_data = []
        self.current_video = None
        self.thumbnail_cache = {}
        if os.path.exists('token.pickle'):
            try:
                os.remove('token.pickle')
            except Exception as e:
                print(f"Error removing token.pickle: {e}")
        self.app_frame.pack_forget()
        self.setup_login_frame()

if __name__ == '__main__':
    root = tk.Tk()
    try:
        app = YouTubeCommentModerator(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application crashed: {str(e)}")
        raise
