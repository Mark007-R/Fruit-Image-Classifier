import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class PaperButton(tk.Button):
    """Flat tk.Button in the mark.dev paper style (presentation only).

    Primary buttons are filled with the accent; the rest are outlined with the paper
    border. Colours are repainted on hover and whenever the state option changes, so a
    disabled primary button reads as disabled instead of as a live accent bar.
    """

    def __init__(self, master, palette, primary=False, **kw):
        self._palette = palette
        self._primary = primary
        self._hovering = False
        super().__init__(master, **kw)
        self.bind('<Enter>', self._on_enter, add='+')
        self.bind('<Leave>', self._on_leave, add='+')
        self._paint()

    def configure(self, cnf=None, **kw):
        result = super().configure(cnf, **kw)
        if 'state' in kw or (isinstance(cnf, dict) and 'state' in cnf):
            self._paint()
        return result

    config = configure

    def _on_enter(self, _event):
        self._hovering = True
        self._paint()

    def _on_leave(self, _event):
        self._hovering = False
        self._paint()

    def _paint(self):
        p = self._palette
        disabled = str(self.cget('state')) == 'disabled'
        if self._primary and disabled:
            bg, fg, edge = p['disabled_bg'], p['disabled_fg'], p['disabled_bg']
        elif self._primary:
            bg = p['accent_strong'] if self._hovering else p['accent']
            fg, edge = p['on_accent'], bg
        elif disabled:
            bg, fg, edge = p['card'], p['disabled_ink'], p['line']
        elif self._hovering:
            bg, fg, edge = p['accent_tint'], p['accent'], p['accent']
        else:
            # outline buttons sit on the alt paper with the strong line, so they read as buttons on a card
            bg, fg, edge = p['paper_alt'], p['ink_2'], p['line_strong']
        tk.Button.configure(
            self, bg=bg, fg=fg, disabledforeground=fg,
            activebackground=p['accent_strong'] if self._primary else p['accent_tint'],
            activeforeground=p['on_accent'] if self._primary else p['accent'],
            highlightbackground=edge, highlightcolor=edge)


class ModernFruitClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Fruit Classifier")
        self.root.geometry("1600x900")
        self.root.configure(bg='#fefaf5')

        self.svm_model = None
        self.dt_model = None
        self.knn_model = None
        self.scaler = StandardScaler()
        self.class_names = []
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.current_image_path = None
        # mark.dev paper palette, Leaf accent. The attribute names are kept from the old
        # dark theme because the methods below refer to them.
        self.bg_primary = '#fefaf5'      # paper ground
        self.bg_secondary = '#faf2e9'    # inset band: stats, image well, readout
        self.bg_card = '#fffdfa'         # cards
        self.accent_purple = '#3B8231'   # Leaf accent: the primary actions (Train, Classify)
        self.accent_blue = '#57504a'     # secondary ink (outline buttons)
        self.accent_green = '#3f7a3a'    # status ok: success text and accuracies
        self.accent_orange = '#a86a12'   # status warn (its button is drawn as an outline)
        self.text_primary = '#1c1714'    # ink
        self.text_secondary = '#756c65'  # muted ink
        self.accent_strong = '#2F6827'
        self.accent_soft = '#62A15A'
        self.line = '#e9dbcd'
        self.line_strong = '#d8c3b2'
        self.palette = {
            'accent': self.accent_purple, 'accent_strong': self.accent_strong,
            'accent_tint': '#f3f6ee', 'on_accent': '#fefaf5',
            'card': self.bg_card, 'paper_alt': self.bg_secondary, 'ink_2': '#57504a',
            'line': self.line, 'line_strong': self.line_strong,
            'disabled_bg': '#e7eee2', 'disabled_fg': '#93b98b', 'disabled_ink': '#b4a79c',
        }
        self.setup_styles()
        self.create_modern_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Modern.Horizontal.TProgressbar",
                       background=self.accent_purple,
                       troughcolor=self.line,
                       borderwidth=0,
                       thickness=4)

    def create_modern_ui(self):
        header = tk.Frame(self.root, bg=self.bg_primary)
        header.pack(fill='x', padx=25, pady=(15, 10))
        tk.Label(header, text="AI Fruit Classifier", font=('Georgia', 24, 'bold'), bg=self.bg_primary, fg=self.text_primary).pack(side='left')
        tk.Label(header, text="Multi-Model ML Classification", font=('Consolas', 10), bg=self.bg_primary, fg=self.accent_purple).pack(side='left', padx=(16, 0))
        main_frame = tk.Frame(self.root, bg=self.bg_primary)
        main_frame.pack(fill='both', expand=True, padx=25, pady=(0, 15))
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        left_col = tk.Frame(main_frame, bg=self.bg_primary)
        left_col.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.create_training_section(left_col)
        middle_col = tk.Frame(main_frame, bg=self.bg_primary)
        middle_col.grid(row=0, column=1, sticky='nsew', padx=(0, 10))
        self.create_upload_section(middle_col)
        right_col = tk.Frame(main_frame, bg=self.bg_primary)
        right_col.grid(row=0, column=2, sticky='nsew')
        self.create_results_section(right_col)

    def create_compact_card(self, parent, title, icon=""):
        card = tk.Frame(parent, bg=self.bg_card, highlightbackground=self.line, highlightcolor=self.line, highlightthickness=1)
        card.pack(fill='both', expand=True, pady=(0, 12))
        header = tk.Frame(card, bg=self.bg_card)
        header.pack(fill='x', padx=16, pady=(12, 8))
        tk.Label(header, text=title, font=('Georgia', 13, 'bold'), bg=self.bg_card, fg=self.text_primary).pack(anchor='w')
        content_frame = tk.Frame(card, bg=self.bg_card)
        content_frame.pack(fill='both', expand=True, padx=16, pady=(0, 14))
        return content_frame

    def create_button(self, parent, text, command, bg_color):
        # Filled accent for the primary actions, paper outline for everything else.
        btn = PaperButton(parent, self.palette, primary=(bg_color == self.accent_purple),
                          text=text, command=command,
                          font=('Segoe UI', 9, 'bold'),
                          relief='flat', padx=15, pady=8, cursor='hand2',
                          borderwidth=0, highlightthickness=1)
        return btn

    def create_training_section(self, parent):
        card1 = self.create_compact_card(parent, "Dataset")
        self.create_button(card1, "Browse Folder", self.select_dataset, self.accent_blue).pack(fill='x', pady=(0, 8))
        self.dataset_label = tk.Label(card1, text="No dataset", font=('Segoe UI', 9), bg=self.bg_card, fg=self.text_secondary, wraplength=280)
        self.dataset_label.pack(fill='x')
        card2 = self.create_compact_card(parent, "Training")
        self.btn_train = self.create_button(card2, "Train Models", self.train_models, self.accent_purple)
        self.btn_train.pack(fill='x', pady=(0, 8))
        self.btn_train.config(state='disabled')
        self.progress = ttk.Progressbar(card2, mode='indeterminate', style="Modern.Horizontal.TProgressbar")
        self.progress.pack(fill='x', pady=(0, 8))
        stats = tk.Frame(card2, bg=self.bg_secondary, highlightbackground=self.line, highlightcolor=self.line, highlightthickness=1, pady=4)
        stats.pack(fill='x')
        self.stats_labels = []
        for model, color in [("SVM", self.text_secondary), ("D-Tree", self.text_secondary), ("KNN", self.text_secondary)]:
            row = tk.Frame(stats, bg=self.bg_secondary)
            row.pack(fill='x', pady=2, padx=8)
            tk.Label(row, text=model, font=('Consolas', 9, 'bold'), bg=self.bg_secondary, fg=color, width=8, anchor='w').pack(side='left')
            label = tk.Label(row, text="—", font=('Consolas', 9), bg=self.bg_secondary, fg=self.text_secondary)
            label.pack(side='right')
            self.stats_labels.append(label)
        card3 = self.create_compact_card(parent, "Models")
        btn_frame = tk.Frame(card3, bg=self.bg_card)
        btn_frame.pack(fill='x')
        self.create_button(btn_frame, "Save", self.save_models, self.accent_green).pack(side='left', expand=True, fill='x', padx=(0, 5))
        self.create_button(btn_frame, "Load", self.load_models, self.accent_orange).pack(side='left', expand=True, fill='x')

    def create_upload_section(self, parent):
        card = self.create_compact_card(parent, "Image Upload")
        self.image_container = tk.Frame(card, bg=self.bg_secondary, highlightbackground=self.line, highlightcolor=self.line, highlightthickness=1)
        self.image_container.pack(fill='both', expand=True, pady=(0, 10))
        self.image_label = tk.Label(self.image_container, text="No Image\n\nUpload to classify", font=('Segoe UI', 11), bg=self.bg_secondary, fg=self.text_secondary)
        self.image_label.pack(expand=True)
        btn_frame = tk.Frame(card, bg=self.bg_card)
        btn_frame.pack(fill='x')
        self.create_button(btn_frame, "Upload", self.upload_image, self.accent_blue).pack(side='left', expand=True, fill='x', padx=(0, 5))
        self.btn_classify = self.create_button(btn_frame, "Classify", self.classify_image, self.accent_purple)
        self.btn_classify.pack(side='left', expand=True, fill='x')
        self.btn_classify.config(state='disabled')

    def create_results_section(self, parent):
        card = self.create_compact_card(parent, "Results")
        self.results_text = tk.Text(card, font=('Consolas', 10), bg=self.bg_secondary, fg=self.text_primary, padx=14, pady=12, wrap='word', relief='flat', borderwidth=0, insertbackground=self.text_primary,
                                    highlightthickness=1, highlightbackground=self.line, highlightcolor=self.line,
                                    selectbackground='#dfe9da', selectforeground=self.text_primary, inactiveselectbackground='#dfe9da')
        self.results_text.pack(fill='both', expand=True)
        self.results_text.insert('1.0', "Awaiting classification...\n\n1. Select dataset\n2. Train models\n3. Upload image\n4. Click classify")
        self.results_text.config(state='disabled')

    # --------- Functional Methods ----------

    def extract_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to load image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (128, 128))
            mean_r, mean_g, mean_b = np.mean(img_resized[:, :, 0]), np.mean(img_resized[:, :, 1]), np.mean(img_resized[:, :, 2])
            std_r, std_g, std_b = np.std(img_resized[:, :, 0]), np.std(img_resized[:, :, 1]), np.std(img_resized[:, :, 2])
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist /= hist.sum()
            texture_mean = np.mean(hist)
            texture_std = np.std(hist)
            texture_energy = np.sum(hist ** 2)
            texture_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (128 * 128)
            return np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b, texture_mean, texture_std, texture_energy, texture_entropy, edge_density])
        except Exception as e:
            logging.error(f"Feature extraction: {e}")
            return None

    def load_dataset(self, dataset_path):
        X, y, class_names = [], [], []
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in class_names:
                class_names.append(class_name)
            class_idx = class_names.index(class_name)
            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                features = self.extract_features(os.path.join(class_path, img_name))
                if features is not None:
                    X.append(features)
                    y.append(class_idx)
        return np.array(X), np.array(y), class_names

    def select_dataset(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path = folder
            try:
                classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
                self.dataset_label.config(text=f"✓ {len(classes)} classes", fg=self.accent_green)
                self.btn_train.config(state='normal')
            except:
                self.dataset_label.config(text="✓ Selected", fg=self.accent_green)
                self.btn_train.config(state='normal')

    def train_models(self):
        if not hasattr(self, 'dataset_path'):
            messagebox.showerror("Error", "No dataset selected.")
            return
        self.progress.start()
        self.btn_train.config(state='disabled')
        self.root.update()
        try:
            X, y, self.class_names = self.load_dataset(self.dataset_path)
            if len(X) == 0:
                raise Exception("No valid data found.")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            self.svm_model.fit(self.X_train, self.y_train)
            svm_acc = accuracy_score(self.y_test, self.svm_model.predict(self.X_test))
            self.dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
            self.dt_model.fit(self.X_train, self.y_train)
            dt_acc = accuracy_score(self.y_test, self.dt_model.predict(self.X_test))
            self.knn_model = KNeighborsClassifier(n_neighbors=5)
            self.knn_model.fit(self.X_train, self.y_train)
            knn_acc = accuracy_score(self.y_test, self.knn_model.predict(self.X_test))
            self.stats_labels[0].config(text=f"{svm_acc*100:.1f}%", fg=self.accent_green)
            self.stats_labels[1].config(text=f"{dt_acc*100:.1f}%", fg=self.accent_green)
            self.stats_labels[2].config(text=f"{knn_acc*100:.1f}%", fg=self.accent_green)
            self.progress.stop()
            self.btn_train.config(state='normal')
            self.btn_classify.config(state='normal')
            logging.info("Models trained successfully.")
            messagebox.showinfo("Success", "Models trained!")
        except Exception as e:
            self.progress.stop()
            self.btn_train.config(state='normal')
            logging.error(f"Training failed: {e}")
            messagebox.showerror("Error", str(e))

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            try:
                self.current_image_path = file_path
                img = Image.open(file_path)
                
                # Calculate dimensions to fit container
                container_width = 400
                container_height = 580
                img.thumbnail((container_width, container_height))
                
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                if self.svm_model is not None:
                    self.btn_classify.config(state='normal')
                
                self.results_text.config(state='normal')
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert('1.0', "Image loaded!\nReady to classify.")
                self.results_text.config(state='disabled')
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def classify_image(self):
        if not self.current_image_path:
            messagebox.showerror("Error", "Upload an image first.")
            return
        if not self.svm_model:
            messagebox.showerror("Error", "Train or load models first.")
            return
        try:
            features = self.extract_features(self.current_image_path)
            if features is None:
                messagebox.showerror("Error", "Feature extraction failed.")
                return
            scaled = self.scaler.transform([features])
            svm_pred = self.svm_model.predict(scaled)[0]
            svm_proba = self.svm_model.predict_proba(scaled)[0]
            dt_pred = self.dt_model.predict(scaled)[0]
            knn_pred = self.knn_model.predict(scaled)[0]
            votes = [svm_pred, dt_pred, knn_pred]
            final_pred = max(set(votes), key=votes.count)
            result = (
                f"SVM: {self.class_names[svm_pred]} ({max(svm_proba)*100:.1f}% confidence)\n"
                f"Decision Tree: {self.class_names[dt_pred]}\n"
                f"K-NN: {self.class_names[knn_pred]}\n"
                f"Final Prediction: {self.class_names[final_pred].upper()} ({votes.count(final_pred)}/3 models agree)\n"
            )
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result)
            self.results_text.config(state='disabled')
            logging.info(f"Image classified as {self.class_names[final_pred]}")
        except Exception as e:
            logging.error(f"Classification failed: {e}")
            messagebox.showerror("Error", str(e))

    def save_models(self):
        if not self.svm_model:
            messagebox.showerror("Error", "Train models first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")])
        if path:
            try:
                with open(path, 'wb') as f:
                    pickle.dump({
                        'svm': self.svm_model,
                        'dt': self.dt_model,
                        'knn': self.knn_model,
                        'scaler': self.scaler,
                        'class_names': self.class_names
                    }, f)
                messagebox.showinfo("Success", "Models saved!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_models(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if path:
            try:
                with open(path, 'rb') as f:
                    models = pickle.load(f)
                    self.svm_model = models['svm']
                    self.dt_model = models['dt']
                    self.knn_model = models['knn']
                    self.scaler = models['scaler']
                    self.class_names = models['class_names']
                
                for label in self.stats_labels:
                    label.config(text="Loaded ✓", fg=self.accent_green)
                
                self.btn_classify.config(state='normal')
                messagebox.showinfo("Success", "Models loaded!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = ModernFruitClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()