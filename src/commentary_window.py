"""
Commentary Display Window
Shows the model's language output in a separate GUI window.
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
from datetime import datetime


class CommentaryWindow:
    """Displays model commentary in a separate GUI window."""
    
    def __init__(self):
        """Initialize the commentary window."""
        self.window = None
        self.text_widget = None
        self.running = False
        self.thread = None
        self.message_queue = queue.Queue()
        self.last_message = None  # Track last message to avoid duplicates
        
    def start(self):
        """Start the commentary window in a separate thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()

        # Wait a bit for window to initialize
        import time
        time.sleep(0.5)
        
    def _run_window(self):
        """Run the tkinter window (called in separate thread)."""
        self.window = tk.Tk()
        self.window.geometry("900x600")
        self.window.configure(bg='#0a0e27')  # Deep blue-black background

        # Header frame
        header_frame = tk.Frame(self.window, bg='#1a1f3a', height=90)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)

        # Title label
        title_label = tk.Label(
            header_frame,
            text="SimLingo AI Commentary",
            font=("Segoe UI", 22, "bold"),
            bg='#1a1f3a',
            fg='#00d4ff'  # Bright cyan
        )
        title_label.pack(pady=(15, 5))

        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Real-time driving decisions and reasoning",
            font=("Segoe UI", 10),
            bg='#1a1f3a',
            fg='#7a8fb5'  # Muted blue-gray
        )
        subtitle_label.pack(pady=(0, 10))

        # Scrolled text widget with modern styling
        self.text_widget = scrolledtext.ScrolledText(
            self.window,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=("Consolas", 14),  # Larger, clearer font
            bg='#0f1419',  # Very dark blue-black
            fg='#e6f1ff',  # Soft white-blue text
            insertbackground='#00d4ff',
            selectbackground='#2d4f67',
            selectforeground='#ffffff',
            borderwidth=0,
            highlightthickness=0,
            padx=15,
            pady=15,
            state='disabled'
        )
        self.text_widget.pack(padx=20, pady=(10, 10), fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = tk.Frame(self.window, bg='#0a0e27')
        button_frame.pack(pady=(0, 15))

        # Clear button with modern styling
        clear_button = tk.Button(
            button_frame,
            text="Clear History",
            command=self._clear_text,
            bg='#1e3a5f',
            fg='#ffffff',
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2',
            activebackground='#2d5a8f',
            activeforeground='#ffffff'
        )
        clear_button.pack()

        # Initial message
        self._append_text("Waiting for AI commentary...\n", color='#7a8fb5')
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start processing message queue
        self._process_queue()

        # Run the window
        self.window.mainloop()
        
    def _on_close(self):
        """Handle window close event."""
        self.running = False
        if self.window:
            self.window.destroy()
            
    def _clear_text(self):
        """Clear all text from the window."""
        if self.text_widget:
            self.text_widget.configure(state='normal')
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.configure(state='disabled')
            
    def _append_text(self, text, color='#e6f1ff'):
        """Append text to the window (thread-safe)."""
        if self.text_widget and self.running:
            try:
                # Skip duplicate messages
                if text == self.last_message:
                    return
                self.last_message = text

                self.text_widget.configure(state='normal')

                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")

                # Add separator line for better readability
                self.text_widget.insert(tk.END, "-" * 100 + "\n", 'separator')

                # Insert timestamp
                self.text_widget.insert(tk.END, f"[{timestamp}]\n", 'timestamp')

                # Clean up text - remove "Waypoints:" suffix if present (it's truncated anyway)
                clean_text = text.replace(" Waypoints:", "").strip()

                # Insert commentary text
                self.text_widget.insert(tk.END, f"{clean_text}\n\n", 'commentary')

                # Configure tags for colors and styling
                self.text_widget.tag_config('separator', foreground='#2d4f67', font=("Consolas", 10))
                self.text_widget.tag_config('timestamp', foreground='#7a8fb5', font=("Consolas", 11, "italic"))
                self.text_widget.tag_config('commentary', foreground='#00d4ff', font=("Consolas", 14, "bold"))

                # Auto-scroll to bottom
                self.text_widget.see(tk.END)

                self.text_widget.configure(state='disabled')
            except:
                pass  # Window might be closing
                
    def _process_queue(self):
        """Process messages from the queue (runs in GUI thread)."""
        try:
            while True:
                text = self.message_queue.get_nowait()
                self._append_text(text)
        except queue.Empty:
            pass

        # Schedule next check
        if self.running and self.window:
            self.window.after(100, self._process_queue)

    def update_commentary(self, text):
        """
        Update the commentary window with new text.

        Args:
            text: Commentary text to display
        """
        if not self.running:
            return

        # Handle empty text
        if not text or text.strip() == "":
            text = "(empty)"

        # Add to queue for processing in GUI thread
        try:
            self.message_queue.put(text)
        except Exception as e:
            print(f"DEBUG: Failed to queue commentary: {e}")
                
    def stop(self):
        """Stop the commentary window."""
        self.running = False
        if self.window:
            try:
                self.window.quit()
            except:
                pass

