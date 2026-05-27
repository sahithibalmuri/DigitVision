import os
import io
import re
import json
import urllib.request
import zipfile
import threading
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------------------------------
# 1. Dataset Downloader & Local Cacher
# ----------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "spam.tsv")

MOCK_DATA = [
    # SPAM samples
    ("spam", "WINNER! As a valued network customer you have been selected to receive a £900 prize reward! Claim now. Call 09061701461. Claim code KL341. Valid 12 hours only."),
    ("spam", "FREE!! Free entry in 2 a weekly draw to win cups, tickets, and cash prizes! Text WIN to 87121 now to claim your reward. Hurry, offer ends soon!"),
    ("spam", "URGENT! Your mobile number has been awarded a £2000 cash bonus! Call 09058094507 now. Claim code 4433. Call costs 150p/min."),
    ("spam", "Congratulations! You have won a 1-week all-expenses-paid trip to Hawaii! Reply 'YES' to claim. No purchase necessary. 18+ only."),
    ("spam", "URGENT NOTICE: Your bank account details need urgent verification. Click the secure link http://secure-bank-login.com to avoid account suspension."),
    ("spam", "Get rich quick! Earn up to $5000 per day working from home. No experience required. Guaranteed payouts. Click here to start now!"),
    ("spam", "Private message: Your invoice is overdue. Please pay £450.50 immediately to avoid legal action. Go to http://pay-invoice-due.com now."),
    ("spam", "Dear customer, your Amazon account has been locked due to suspicious activity. Verify your identity at http://amazon-security-alert.com."),
    ("spam", "Double your money in just 24 hours! Safe and secured cryptocurrency investment. WhatsApp us at +12345678 to double your wallet."),
    ("spam", "Weekly special: Extra 20% off on all male health products and pills. Order online today. Discreet shipping. Buy now at discount prices!"),
    ("spam", "Get a brand new iPhone 15 for only £1! Limited stock available. Spin the wheel to claim your reward. Click http://win-new-phone.com."),
    ("spam", "HOT deals! Meet singles in your local area tonight. 100% anonymous chat. Text CHAT to 69888 now. Standard network rates apply."),
    ("spam", "IMPORTANT Account Notification: Please update your credit card details immediately to keep your Netflix subscription active at http://netflix-update.com."),
    ("spam", "Cash cash cash! Need an instant personal loan up to $5000? No credit check. Approved in 5 minutes. Apply at http://instant-cash-loans.com."),
    ("spam", "You have 1 new voicemail. To listen to your message, call 09099726395. Rate 200p/min. Reply STOP to opt out."),
    ("spam", "Final chance! Get 80% off luxury Swiss watches. Free delivery worldwide. Perfect gift. Buy today at http://swiss-luxury-watches.com."),
    ("spam", "Verify your PayPal wallet instantly. Click here to confirm your email and unlock restricted funds at http://paypal-verify-wallet.com."),
    ("spam", "Exclusive invitation: Join the VIP online casino today and get a $200 free signup bonus! Spin to win real cash rewards now."),
    ("spam", "Get cheap flights! Book your summer holidays today and save up to 50% on major airlines. Visit http://cheap-flights-deal.com."),
    ("spam", "Congratulations! Your CV has been shortlisted for a high-paying executive assistant position. WhatsApp us at +19876543 to schedule an interview."),
    ("spam", "Dear User, your iCloud storage is almost full. Upgrade today or risk losing your photos. Go to http://icloud-storage-full.com."),
    ("spam", "Earn passive income by renting out your unused digital storage. Earn up to $100/week. Clean, safe, and automated. Sign up now!"),
    ("spam", "Your package delivery failed. Courier could not find your address. Update your shipping details and pay £1.50 re-delivery fee at http://post-office-delivery.com."),
    ("spam", "Urgent security update required for your Microsoft Account. Please click the link to confirm your password: http://microsoft-account-secure.com."),
    ("spam", "Special promotional offer! Get a lifetime subscription to our Premium VPN for a one-time fee of $19. Limited time only. Click here."),

    # HAM samples
    ("ham", "Hi, are we still on for lunch today? I was thinking we could try that new Italian place down the street. Let me know if that works!"),
    ("ham", "Hey, just wanted to check if you finished the slides for tomorrow's project meeting? I'm putting the final presentation together now."),
    ("ham", "Dear student, please note that the deadline for submitting the course assignment has been extended to Friday at 5:00 PM. Best regards."),
    ("ham", "Hi mom, I will be a bit late coming home tonight. Got caught up with some extra work at the office. Don't wait up for dinner!"),
    ("ham", "Thanks for the quick response. I've scheduled the calendar invite for our sync tomorrow at 10 AM. Let me know if you need to reschedule."),
    ("ham", "Hey! Can you send me the recipe for that delicious chocolate cake you made last weekend? My sister wants to bake it for her birthday."),
    ("ham", "Hello, your package has been successfully delivered to the reception desk. Please pick it up at your convenience. Thank you!"),
    ("ham", "Hi Team, please find attached the minutes of today's review meeting. Let me know if there are any corrections or comments before I publish them."),
    ("ham", "Hey buddy, are you watching the game tonight? We are gathering at Mark's place around 7 PM. Let me know if you want to join."),
    ("ham", "Good morning! Just a friendly reminder that your dentist appointment is scheduled for tomorrow at 2:30 PM. Please reply to confirm."),
    ("ham", "Hello, I hope you are having a productive week. I reviewed your draft document and made a few minor edits. It looks really solid!"),
    ("ham", "Hi, sorry I missed your call earlier. I was in a lecture. I'll call you back as soon as I get home around 6 PM."),
    ("ham", "Hey, do you have the phone number of the plumber who fixed your sink last month? Our bathroom pipe is leaking and we need help fast."),
    ("ham", "Hi Sarah, just wanted to say happy birthday! Hope you have an amazing day filled with joy and lots of cake. Talk to you soon!"),
    ("ham", "Dear team member, please submit your weekly timesheets by today evening to ensure smooth payroll processing. Thank you."),
    ("ham", "Hey, I left my blue folder on your desk after the afternoon seminar. Can you please keep it safe for me? I'll grab it tomorrow."),
    ("ham", "Dear customer, your monthly billing statement for May is now available online. Log in to your account portal to view the details."),
    ("ham", "Hi, are you free this weekend? We were planning a small hiking trip to the hills on Sunday morning. Let me know if you can make it."),
    ("ham", "Hey, I checked the store and they don't have the book you wanted in stock. I can order it online for you, it should arrive by Wednesday."),
    ("ham", "Hello, this is to confirm that we received your application for the Software Engineer role. Our HR team will review it shortly."),
    ("ham", "Hi Jack, the car service is done and it's ready for collection. Total cost is £120. We are open until 6 PM today. Thanks!"),
    ("ham", "Hey, just landed in London! The flight was a bit bumpy but we made it. Heading to the hotel now. Talk to you later!"),
    ("ham", "Hi, can we reschedule our meeting to 4 PM today? A sudden urgent client request came up that I need to resolve immediately."),
    ("ham", "Hey! Let's meet at the coffee shop near our old university. It's been ages since we last caught up. Looking forward to it!"),
    ("ham", "Hello, thank you for participating in our research survey. Your feedback is highly valuable to us. Have a wonderful day.")
]

def load_or_download_dataset():
    """Load the dataset from cache, download from UCI mirror, or fall back to high-quality mocks."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Try loading cached dataset
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'message'], encoding='utf-8')
            if not df.empty:
                print(f"Loaded cached dataset from {DATA_PATH} ({len(df)} samples)")
                return df
        except Exception as e:
            print(f"Failed to load cached dataset: {e}. Re-downloading.")

    # 2. Try downloading from raw github mirror (very fast and clean)
    urls = [
        "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv",
        "https://raw.githubusercontent.com/sahithibalmuri/DigitVision/main/sms.tsv" # fallback mirror
    ]
    
    for url in urls:
        try:
            print(f"Attempting to download dataset from {url}...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('utf-8')
                
            # Quick sanity check
            if "ham\t" in content or "spam\t" in content:
                with open(DATA_PATH, "w", encoding="utf-8") as f:
                    f.write(content)
                df = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'message'], encoding='utf-8')
                print(f"Dataset downloaded successfully: {len(df)} samples cached.")
                return df
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    # 3. Network failed: write and return high-quality mock data
    print("Network download unavailable. Generating premium mock dataset fallback...")
    try:
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            for label, msg in MOCK_DATA:
                f.write(f"{label}\t{msg}\n")
        
        df = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'message'], encoding='utf-8')
        print(f"Created fallback mock dataset with {len(df)} samples.")
        return df
    except Exception as e:
        print(f"Failed to write mock dataset: {e}")
        # Return directly from memory
        return pd.DataFrame(MOCK_DATA, columns=['label', 'message'])


# ----------------------------------------------------
# 2. Machine Learning Pipeline & Helper Functions
# ----------------------------------------------------
class SpamClassifierPipeline:
    def __init__(self, model_type="naive_bayes", vectorizer_params=None, model_params=None):
        self.model_type = model_type
        
        # Default Vectorizer Params
        v_params = {
            "max_features": 2500,
            "stop_words": "english",
            "ngram_range": (1, 1),
            "use_idf": True
        }
        if vectorizer_params:
            v_params.update(vectorizer_params)
            
        self.vectorizer = TfidfVectorizer(
            max_features=v_params["max_features"],
            stop_words=v_params["stop_words"] if v_params["stop_words"] != "none" else None,
            ngram_range=v_params["ngram_range"],
            use_idf=v_params["use_idf"]
        )
        
        # Instantiate Classifier
        if model_type == "svm":
            m_params = {"C": 1.0, "kernel": "linear"}
            if model_params:
                m_params.update(model_params)
            self.model = SVC(
                C=m_params["C"],
                kernel=m_params["kernel"],
                probability=True,  # Enable probability mapping
                random_state=42
            )
        else:  # Naive Bayes
            m_params = {"alpha": 1.0}
            if model_params:
                m_params.update(model_params)
            self.model = MultinomialNB(alpha=m_params["alpha"])
            
        self.is_trained = False
        self.metrics = {}
        self.top_spam_words = []
        self.top_ham_words = []

    def train_and_evaluate(self, df, test_size=0.25):
        """Train the classifier pipeline and extract complete analytics."""
        X = df['message']
        y = df['label'].map({'ham': 0, 'spam': 1})
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Fit vectorizer & transform
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        
        # Predict & Evaluate
        y_pred = self.model.predict(X_test_vec)
        y_prob = self.model.predict_proba(X_test_vec)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        # Extract Keyword Importances
        self._extract_top_words()
        return self.metrics

    def _extract_top_words(self):
        """Extract top spam and ham predictive words using mathematical weights."""
        feature_names = self.vectorizer.get_feature_names_out()
        vocab_size = len(feature_names)
        
        if vocab_size == 0:
            self.top_spam_words = []
            self.top_ham_words = []
            return
            
        if self.model_type == "naive_bayes":
            # Difference in class log probabilities
            log_prob_spam = self.model.feature_log_prob_[1]
            log_prob_ham = self.model.feature_log_prob_[0]
            scores = log_prob_spam - log_prob_ham
        elif self.model_type == "svm" and self.model.kernel == "linear":
            # Linear SVM coefficients
            scores = self.model.coef_[0].toarray()[0] if hasattr(self.model.coef_, "toarray") else self.model.coef_[0]
        else:
            # SVM RBF or fallback: run prediction on individual words to evaluate impact
            # Take a sample of the top 300 most frequent words in training vocabulary
            # (which corresponds to highest TF-IDF averages)
            # To be efficient, we only predict for those
            scores = np.zeros(vocab_size)
            # Find high frequency indexes
            idf_weights = self.vectorizer.idf_
            sorted_freq_indices = np.argsort(idf_weights)[:min(300, vocab_size)]
            
            # Predict single-word documents
            single_words = [feature_names[i] for i in sorted_freq_indices]
            vec_words = self.vectorizer.transform(single_words)
            probs = self.model.predict_proba(vec_words)[:, 1] # spam probability
            
            for idx, f_idx in enumerate(sorted_freq_indices):
                # Scale between -1 and 1
                scores[f_idx] = (probs[idx] - 0.5) * 2.0

        # Sort indices
        sorted_indices = np.argsort(scores)
        
        # Ham words (negative weights / higher ham log prob)
        top_ham_indices = sorted_indices[:15]
        self.top_ham_words = [
            {"word": str(feature_names[i]), "score": float(scores[i])}
            for i in top_ham_indices
        ]
        
        # Spam words (positive weights / higher spam log prob)
        top_spam_indices = sorted_indices[::-1][:15]
        self.top_spam_words = [
            {"word": str(feature_names[i]), "score": float(scores[i])}
            for i in top_spam_indices
        ]

    def predict_message(self, text):
        """Predict spam probability and return token-by-token perturbation breakdown."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        # Predict overall probability
        vec_text = self.vectorizer.transform([text])
        prob_spam = float(self.model.predict_proba(vec_text)[0][1])
        label = "spam" if prob_spam >= 0.5 else "ham"
        confidence = prob_spam if label == "spam" else (1.0 - prob_spam)
        
        # Tokenize preserving spaces and punctuation
        raw_tokens = [tok for tok in re.split(r'(\s+|[^\w\s\-\'])', text) if tok]
        
        # Find unique words
        word_scores = {}
        words = set(tok.lower() for tok in raw_tokens if re.match(r'^[a-zA-Z0-9\-\']+$', tok))
        
        # Perturbation analysis: remove each word and recalculate spam probability
        for w in words:
            modified_tokens = [tok for tok in raw_tokens if tok.lower() != w]
            modified_text = "".join(modified_tokens)
            
            if not modified_text.strip():
                # If removing the word empties the text, score is the offset from baseline 0.5
                word_scores[w] = prob_spam - 0.5
                continue
                
            mod_vec = self.vectorizer.transform([modified_text])
            mod_prob = float(self.model.predict_proba(mod_vec)[0][1])
            
            # Score represents contribution to SPAM class:
            # If removing word 'w' DECREASES the spam probability, it contributed POSITIVELY to spam (spammy).
            # If removing word 'w' INCREASES the spam probability, it contributed NEGATIVELY to spam (hammy).
            word_scores[w] = prob_spam - mod_prob
            
        # Build token contribution structure
        tokens_breakdown = []
        for tok in raw_tokens:
            is_word = bool(re.match(r'^[a-zA-Z0-9\-\']+$', tok))
            w_lower = tok.lower()
            score = word_scores.get(w_lower, 0.0) if is_word else 0.0
            
            tokens_breakdown.append({
                "token": tok,
                "is_word": is_word,
                "score": score
            })
            
        return {
            "label": label,
            "probability": prob_spam,
            "confidence": confidence,
            "tokens": tokens_breakdown
        }


# ----------------------------------------------------
# 3. Asynchronous Training Controller
# ----------------------------------------------------
class TrainingController:
    def __init__(self):
        self.is_training = False
        self.status_message = "Idle"
        self.console_logs = []
        self.lock = threading.Lock()
        
        # In-memory models and vectorizers
        self.pipelines = {
            "naive_bayes": None,
            "svm": None
        }
        self.current_model = "naive_bayes"
        
        # Hyperparameters
        self.hyperparameters = {
            "model_type": "naive_bayes",
            "max_features": 2500,
            "stop_words": "english",
            "ngram_range_max": 1, # 1 for unigrams, 2 for bigrams
            "use_idf": True,
            "split_ratio": 0.25, # test size
            "nb_alpha": 1.0,
            "svm_c": 1.0,
            "svm_kernel": "linear",
            "subset_size": 0 # 0 means full dataset
        }
        
        self.logs_lock = threading.Lock()

    def add_log(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        formatted = f"{timestamp} {message}"
        print(f"SpamClassifier Training: {message}")
        with self.logs_lock:
            self.console_logs.append(formatted)
            # Limit logs size
            if len(self.console_logs) > 200:
                self.console_logs.pop(0)

    def get_status(self):
        with self.lock:
            nb_pipe = self.pipelines["naive_bayes"]
            svm_pipe = self.pipelines["svm"]
            
            # Fetch status info
            nb_stats = nb_pipe.metrics if (nb_pipe and nb_pipe.is_trained) else None
            svm_stats = svm_pipe.metrics if (svm_pipe and svm_pipe.is_trained) else None
            
            # Top words of currently active model
            active_pipe = self.pipelines[self.current_model]
            top_spam = active_pipe.top_spam_words if (active_pipe and active_pipe.is_trained) else []
            top_ham = active_pipe.top_ham_words if (active_pipe and active_pipe.is_trained) else []
            
            return {
                "is_training": self.is_training,
                "status_message": self.status_message,
                "console_logs": self.console_logs,
                "current_model": self.current_model,
                "hyperparameters": self.hyperparameters,
                "models_trained": {
                    "naive_bayes": nb_pipe is not None and nb_pipe.is_trained,
                    "svm": svm_pipe is not None and svm_pipe.is_trained
                },
                "metrics": {
                    "naive_bayes": nb_stats,
                    "svm": svm_stats
                },
                "top_spam_words": top_spam,
                "top_ham_words": top_ham
            }

    def start_training(self, params):
        with self.lock:
            if self.is_training:
                return False, "Training already in progress."
                
            self.is_training = True
            self.status_message = "Training scheduled..."
            self.hyperparameters.update(params)
            
            # Spawn training thread
            thread = threading.Thread(target=self._run_training)
            thread.daemon = True
            thread.start()
            return True, "Training started."

    def _run_training(self):
        try:
            self.add_log("Initializing background training thread...")
            
            # Read current configurations
            model_type = self.hyperparameters["model_type"]
            max_features = int(self.hyperparameters["max_features"])
            stop_words = self.hyperparameters["stop_words"]
            ngram_max = int(self.hyperparameters["ngram_range_max"])
            use_idf = bool(self.hyperparameters["use_idf"])
            split_ratio = float(self.hyperparameters["split_ratio"])
            subset_size = int(self.hyperparameters["subset_size"])
            
            nb_alpha = float(self.hyperparameters["nb_alpha"])
            svm_c = float(self.hyperparameters["svm_c"])
            svm_kernel = self.hyperparameters["svm_kernel"]
            
            ngram_range = (1, ngram_max)
            
            self.add_log(f"Loading/Syncing text spam dataset...")
            df = load_or_download_dataset()
            
            with self.lock:
                self.status_message = "Preprocessing text data..."
                
            # Apply subset if configured
            if subset_size > 0 and subset_size < len(df):
                self.add_log(f"Sampling subset of {subset_size} items from {len(df)} total items.")
                df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
            else:
                self.add_log(f"Using full dataset of {len(df)} samples.")
                
            # Log hyperparameters details
            self.add_log(f"Training parameters applied:")
            self.add_log(f" - Model Type: {model_type.upper()}")
            self.add_log(f" - Vectorizer Max Features: {max_features}")
            self.add_log(f" - Stop Words Filter: {stop_words}")
            self.add_log(f" - N-gram Configuration: {ngram_range}")
            self.add_log(f" - TF-IDF Weighting Enabled: {use_idf}")
            self.add_log(f" - Train/Test Split Ratio: {1.0-split_ratio:.2f}/{split_ratio:.2f}")
            
            if model_type == "naive_bayes":
                self.add_log(f" - Naive Bayes Smoothing Alpha: {nb_alpha}")
                vec_params = {
                    "max_features": max_features,
                    "stop_words": stop_words,
                    "ngram_range": ngram_range,
                    "use_idf": use_idf
                }
                model_params = {"alpha": nb_alpha}
            else:
                self.add_log(f" - SVM Regularization C: {svm_c}")
                self.add_log(f" - SVM Kernel Mode: {svm_kernel}")
                vec_params = {
                    "max_features": max_features,
                    "stop_words": stop_words,
                    "ngram_range": ngram_range,
                    "use_idf": use_idf
                }
                model_params = {"C": svm_c, "kernel": svm_kernel}

            self.add_log("Fitting TF-IDF text representation vectorizer...")
            pipeline = SpamClassifierPipeline(
                model_type=model_type,
                vectorizer_params=vec_params,
                model_params=model_params
            )
            
            with self.lock:
                self.status_message = f"Training {model_type.upper()} classifier..."
            
            self.add_log("Fitting classifier model to training vectors...")
            t0 = time.time()
            metrics = pipeline.train_and_evaluate(df, test_size=split_ratio)
            t1 = time.time()
            
            self.add_log(f"Model training successfully completed in {t1-t0:.3f} seconds.")
            self.add_log(f"Validation statistics calculated:")
            self.add_log(f" - Accuracy:  {metrics['accuracy']:.4%}")
            self.add_log(f" - Precision: {metrics['precision']:.4%}")
            self.add_log(f" - Recall:    {metrics['recall']:.4%}")
            self.add_log(f" - F1-Score:  {metrics['f1_score']:.4%}")
            
            cm = metrics['confusion_matrix']
            self.add_log(f" - Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
            
            with self.lock:
                self.pipelines[model_type] = pipeline
                self.current_model = model_type
                self.is_training = False
                self.status_message = f"Training successful! Current model: {model_type.upper()}"
                
            self.add_log(f"Active inference engine updated to {model_type.upper()}. Ready for client scans.")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.add_log(f"TRAINING CRITICAL FAILURE: {str(e)}")
            print(tb)
            with self.lock:
                self.is_training = False
                self.status_message = f"Error during training: {str(e)}"


# Instantiate singleton training controller
controller = TrainingController()


# ----------------------------------------------------
# 4. HTTP Web Request Handler & API Endpoints
# ----------------------------------------------------
class SpamClassifierRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging in console to keep terminal tidy unless error
        pass

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # REST API: Training Status
        if path == "/api/train/status":
            self.send_json(controller.get_status())
            return
            
        # REST API: Preset Sample Emails
        if path == "/api/samples":
            samples = [
                {
                    "title": "Lottery Jackpot Promo (Spam)",
                    "text": "URGENT! You have won a guaranteed cash prize of £2,000,000 in our international weekly lottery draw! To claim your jackpot reward, text WINNER to 88990 immediately or call 0906-883-9920. Entry is completely free! Claim code: X902. Terms and conditions apply, must be 18+ to claim.",
                    "type": "spam"
                },
                {
                    "title": "Security Account Alert (Spam)",
                    "text": "Dear customer, we detected unusual login activity on your PayPal account from an unrecognized IP address in Russia. For your safety, we have temporarily restricted your wallet features. To restore access, please click the secure link immediately: http://paypal-identity-secure-login.com/restore/verify. Failure to do so within 24 hours may result in permanent suspension.",
                    "type": "spam"
                },
                {
                    "title": "Corporate Meeting Scheduling (Ham)",
                    "text": "Hi Team, I hope you're having a productive week. Let's schedule a brief sync tomorrow afternoon at 2 PM to go over the final feedback from the client on the project UI drafts. I've attached the latest mockup file for your review. Let me know if that time works or if we should reschedule to Thursday morning.",
                    "type": "ham"
                },
                {
                    "title": "Casual Weekend Hiking (Ham)",
                    "text": "Hey buddy! Are you free this coming Sunday? A few of us from the office are planning a small hiking trip up the green trails near the valley. We'll start around 7:30 AM to catch the fresh air, have a small picnic lunch, and return by mid-afternoon. Let me know if you can join, we'd love to have you!",
                    "type": "ham"
                }
            ]
            self.send_json(samples)
            return

        # Serve Static Frontend Files
        if path == "/" or path == "":
            path = "/spam.html"
            
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        filepath = os.path.join(static_dir, path.lstrip("/"))
        
        # Prevent Directory Traversal
        if not filepath.startswith(os.path.abspath(static_dir)):
            self.send_error(403, "Access Forbidden")
            return
            
        if os.path.exists(filepath) and os.path.isfile(filepath):
            content_type = "text/plain"
            if filepath.endswith(".html"):
                content_type = "text/html"
            elif filepath.endswith(".css"):
                content_type = "text/css"
            elif filepath.endswith(".js"):
                content_type = "application/javascript"
            elif filepath.endswith(".png"):
                content_type = "image/png"
            elif filepath.endswith(".ico"):
                content_type = "image/x-icon"
                
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Read post body
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            body = json.loads(post_data) if post_data else {}
        except Exception:
            body = {}
            
        # REST API: Start background training
        if path == "/api/train/start":
            success, msg = controller.start_training(body)
            self.send_json({"success": success, "message": msg})
            return
            
        # REST API: Scan email message (Inference)
        if path == "/api/predict":
            text = body.get("text", "").strip()
            model_type = body.get("model", controller.current_model)
            
            if not text:
                self.send_json({"error": "Missing or empty text content"}, 400)
                return
                
            pipeline = controller.pipelines.get(model_type)
            if not pipeline or not pipeline.is_trained:
                self.send_json({"error": f"The requested model ({model_type.upper()}) is not trained yet. Please train it first."}, 400)
                return
                
            try:
                # Set active model if valid inference requested on it
                with controller.lock:
                    controller.current_model = model_type
                result = pipeline.predict_message(text)
                self.send_json({"success": True, "prediction": result})
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, 500)
            return

        self.send_error(404, "API Endpoint Not Found")

    def send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


# ----------------------------------------------------
# 5. Main Entrypoint & Background Initialization
# ----------------------------------------------------
def start_server(port=5000):
    # Ensure static directory exists
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)
    
    server_address = ('', port)
    try:
        httpd = HTTPServer(server_address, SpamClassifierRequestHandler)
        print(f"SpamGuard AI Server running at: http://localhost:{port}/")
        
        # Trigger silent auto-training thread for Naive Bayes on startup
        # To make it immediately usable out-of-the-box!
        # We limit features to 1500 and train size to 3000 to complete in < 0.2 seconds.
        print("Pre-training default Naive Bayes model on background thread...")
        controller.start_training({
            "model_type": "naive_bayes",
            "max_features": 1500,
            "stop_words": "english",
            "ngram_range_max": 1,
            "use_idf": True,
            "split_ratio": 0.25,
            "nb_alpha": 1.0,
            "subset_size": 3000
        })
        
        httpd.serve_forever()
    except OSError as e:
        # Address already in use
        if e.errno == 10048 or "Address already in use" in str(e):
            print(f"Port {port} is occupied. Attempting port {port + 1}...")
            start_server(port + 1)
        else:
            raise e


if __name__ == '__main__':
    start_server()
