import os
import io
import base64
import json
import threading
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

# ----------------------------------------------------
# 1. PyTorch CNN Model Architecture
# ----------------------------------------------------
class MNISTCNN(nn.Module):
    def __init__(self, channels1=16, channels2=32, fc1_units=128):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, channels1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels2)
        
        # MNIST input is 28x28. 
        # After first pooling (2x2): 14x14
        # After second pooling (2x2): 7x7
        self.fc1 = nn.Linear(channels2 * 7 * 7, fc1_units)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(fc1_units, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_feature_maps(self, x):
        """Extract intermediate activations of conv1 and conv2 layers."""
        conv1_out = torch.relu(self.bn1(self.conv1(x)))
        pool1_out = self.pool(conv1_out)
        conv2_out = torch.relu(self.bn2(self.conv2(pool1_out)))
        return conv1_out, conv2_out


# ----------------------------------------------------
# 2. Asynchronous Training Controller
# ----------------------------------------------------
class TrainingController:
    def __init__(self):
        self.is_training = False
        self.stop_requested = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.loss_history = []
        self.accuracy_history = []
        self.status_message = "Idle"
        self.lock = threading.Lock()
        self.model_path = "mnist_cnn.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.hyperparameters = {
            "lr": 0.001,
            "optimizer": "Adam",
            "epochs": 5,
            "batch_size": 64,
            "complexity": "Medium",
            "train_size": 20000  # Default to a subset for fast responsive training
        }
        
        # Load or initialize model
        self.load_model()

    def load_model(self):
        with self.lock:
            # Determine channel counts based on complexity
            c1, c2, fc = self._get_complexity_params(self.hyperparameters["complexity"])
            self.model = MNISTCNN(channels1=c1, channels2=c2, fc1_units=fc).to(self.device)
            
            if os.path.exists(self.model_path):
                try:
                    # Allow mapping to CPU if CUDA is not available
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    # Create temporary model to check shape compatibility
                    temp_model = MNISTCNN(channels1=16, channels2=32, fc1_units=128).to(self.device)
                    temp_model.load_state_dict(state_dict, strict=False)
                    # If loaded successfully, rebuild model with proper complexity
                    # Wait, MNIST weights shape might differ. We check shapes before assigning.
                    self.model = temp_model
                    self.status_message = "Pre-trained model loaded successfully."
                    print("Loaded pre-trained model.")
                except Exception as e:
                    self.status_message = f"Error loading model: {str(e)}. Reinitializing."
                    print(f"Failed to load model: {e}")
            else:
                self.status_message = "No model found. Ready to train."

    def _get_complexity_params(self, complexity):
        if complexity == "Small":
            return 8, 16, 64
        elif complexity == "Large":
            return 32, 64, 256
        else:  # Medium
            return 16, 32, 128

    def get_status(self):
        with self.lock:
            return {
                "is_training": self.is_training,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "loss_history": self.loss_history,
                "accuracy_history": self.accuracy_history,
                "status_message": self.status_message,
                "device": str(self.device),
                "hyperparameters": self.hyperparameters
            }

    def stop_training(self):
        with self.lock:
            if self.is_training:
                self.stop_requested = True
                self.status_message = "Stop requested. Wrapping up..."

    def start_training(self, params):
        with self.lock:
            if self.is_training:
                return False, "Training is already in progress."
            
            self.is_training = True
            self.stop_requested = False
            self.current_epoch = 0
            self.loss_history = []
            self.accuracy_history = []
            self.hyperparameters.update(params)
            self.total_epochs = int(self.hyperparameters["epochs"])
            
            # Rebuild model according to selected complexity
            c1, c2, fc = self._get_complexity_params(self.hyperparameters["complexity"])
            self.model = MNISTCNN(channels1=c1, channels2=c2, fc1_units=fc).to(self.device)
            
            # Spawn background training thread
            thread = threading.Thread(target=self._run_training)
            thread.daemon = True
            thread.start()
            return True, "Training started in background."

    def _run_training(self):
        try:
            self._execute_training()
        except Exception as e:
            with self.lock:
                self.is_training = False
                self.status_message = f"Training failed: {str(e)}"
                print(f"Training error: {e}")

    def _execute_training(self):
        with self.lock:
            self.status_message = "Preparing MNIST dataset..."
            lr = float(self.hyperparameters["lr"])
            opt_name = self.hyperparameters["optimizer"]
            batch_size = int(self.hyperparameters["batch_size"])
            train_subset_size = int(self.hyperparameters["train_size"])
        
        # Prepare transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download datasets
        print("Downloading/Loading MNIST dataset...")
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Use subsets for faster epochs if specified
        if train_subset_size < len(train_dataset):
            # Deterministic subset for reproducibility
            indices = list(range(train_subset_size))
            train_dataset = Subset(train_dataset, indices)
            
        # Test subset to keep evaluation fast
        test_dataset = Subset(test_dataset, list(range(2000)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # Select optimizer
        if opt_name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else: # Adam
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, self.total_epochs + 1):
            with self.lock:
                if self.stop_requested:
                    self.status_message = "Training stopped by user."
                    break
                self.status_message = f"Training epoch {epoch}/{self.total_epochs}..."
                self.current_epoch = epoch

            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                with self.lock:
                    if self.stop_requested:
                        break
                
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            if self.stop_requested:
                break
                
            epoch_loss = epoch_loss / total
            train_acc = 100. * correct / total
            
            # Evaluate on validation subset
            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            # Record metrics
            with self.lock:
                self.loss_history.append(epoch_loss)
                self.accuracy_history.append(val_acc)
                print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        with self.lock:
            self.is_training = False
            if not self.stop_requested:
                # Save trained model weights
                torch.save(self.model.state_dict(), self.model_path)
                self.status_message = f"Training complete! Model saved. Final Val Acc: {val_acc:.2f}%"
                print(f"Saved model with Val Acc {val_acc:.2f}% to {self.model_path}")
            self.stop_requested = False


# Create training controller instance
controller = TrainingController()


# ----------------------------------------------------
# 3. HTTP Server and Request Handler
# ----------------------------------------------------
class DigitVisionRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP logging to keep console clean unless it's an error
        pass

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # API Route: Fetch training status
        if path == "/api/train/status":
            self.send_json(controller.get_status())
            return
            
        # API Route: Model info
        if path == "/api/model/info":
            info = {
                "architecture": "MNISTCNN (2x Conv, 2x MaxPool, 2x FullyConnected)",
                "device": str(controller.device),
                "has_weights": os.path.exists(controller.model_path),
                "hyperparameters": controller.hyperparameters
            }
            self.send_json(info)
            return

        # Serve static files
        if path == "/" or path == "":
            path = "/index.html"
            
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        filepath = os.path.join(static_dir, path.lstrip("/"))
        
        # Security check: Ensure file is inside static_dir
        if not filepath.startswith(os.path.abspath(static_dir)):
            self.send_error(403, "Access Denied")
            return
            
        if os.path.exists(filepath) and os.path.isfile(filepath):
            # Deduce Content-Type
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
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            body = json.loads(post_data) if post_data else {}
        except Exception:
            body = {}

        # API Route: Inferences / Predictions
        if path == "/api/predict":
            if "image" not in body:
                self.send_json({"error": "Missing image data"}, 400)
                return
                
            # Run inference
            result = self.process_inference(body["image"])
            self.send_json(result)
            return

        # API Route: Start training
        if path == "/api/train/start":
            success, msg = controller.start_training(body)
            self.send_json({"success": success, "message": msg})
            return

        # API Route: Stop training
        if path == "/api/train/stop":
            controller.stop_training()
            self.send_json({"success": True, "message": "Stop request submitted."})
            return

        self.send_error(404, "API Endpoint Not Found")

    def send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def process_inference(self, base64_image_str):
        try:
            # Decode base64 image (stripping the data:image/png;base64, prefix if present)
            if "," in base64_image_str:
                base64_image_str = base64_image_str.split(",")[1]
            image_data = base64.b64decode(base64_image_str)
            
            # Load PIL Image
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to Grayscale
            img_gray = img.convert("L")
            
            # Resize to MNIST standard (28x28) using high-quality downsampling
            img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Preprocess tensor to match MNIST mean & std normalization
            # ToTensor divides by 255.0 to scale to [0,1]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            tensor_in = transform(img_resized).unsqueeze(0).to(controller.device)
            
            # Run forward pass
            controller.model.eval()
            with torch.no_grad():
                outputs = controller.model(tensor_in)
                probabilities = torch.softmax(outputs, dim=1).squeeze(0)
                
                # Get intermediate activations
                conv1_act, conv2_act = controller.model.get_feature_maps(tensor_in)
                
            # Prep predictions list
            preds = probabilities.tolist()
            top_prediction = int(probabilities.argmax().item())
            
            # Generate feature maps base64 grids (8 from layer 1, 8 from layer 2)
            fmaps_l1 = self.extract_fmaps_as_base64(conv1_act, max_maps=8)
            fmaps_l2 = self.extract_fmaps_as_base64(conv2_act, max_maps=8)
            
            return {
                "success": True,
                "predictions": preds,
                "top_digit": top_prediction,
                "feature_maps_l1": fmaps_l1,
                "feature_maps_l2": fmaps_l2
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def extract_fmaps_as_base64(self, tensor_activation, max_maps=8):
        """Extract first N feature maps from layer activation tensor, convert to base64."""
        # tensor shape: [1, num_channels, H, W]
        act_maps = tensor_activation.squeeze(0).cpu() # [num_channels, H, W]
        num_channels = act_maps.size(0)
        
        base64_maps = []
        
        for i in range(min(num_channels, max_maps)):
            fmap = act_maps[i].numpy()
            
            # Min-Max Scaling to 0-255
            fmap_min = fmap.min()
            fmap_max = fmap.max()
            if fmap_max > fmap_min:
                fmap_norm = (fmap - fmap_min) / (fmap_max - fmap_min) * 255.0
            else:
                fmap_norm = fmap * 0.0
                
            fmap_norm = fmap_norm.astype('uint8')
            
            # Convert to PIL Image and resize so it's clearly visible
            img = Image.fromarray(fmap_norm, mode='L')
            # Resize with NEAREST to preserve pixel boundaries (looks highly technical/cool!)
            img_large = img.resize((112, 112), Image.Resampling.NEAREST)
            
            # Encode to PNG Base64
            buffered = io.BytesIO()
            img_large.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_maps.append("data:image/png;base64," + img_str)
            
        return base64_maps


# ----------------------------------------------------
# 4. Main Server Startup
# ----------------------------------------------------
def start_server(port=5000):
    # Ensure static directory exists
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)
    
    server_address = ('', port)
    try:
        httpd = HTTPServer(server_address, DigitVisionRequestHandler)
        print(f"DigitVision Server running at: http://localhost:{port}/")
        
        # Trigger silent auto-training thread if weights don't exist
        if not os.path.exists(controller.model_path):
            print("No pre-trained weights found. Starting auto-train sequence on background thread...")
            # We train with small fast size for instant responsiveness (e.g. 5000 images, 1 epoch)
            controller.start_training({
                "lr": 0.001,
                "optimizer": "Adam",
                "epochs": 1,
                "batch_size": 64,
                "complexity": "Medium",
                "train_size": 5000
            })
            
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 10048 or "Address already in use" in str(e):
            print(f"Port {port} is occupied. Attempting port {port + 1}...")
            start_server(port + 1)
        else:
            raise e


if __name__ == '__main__':
    start_server()
