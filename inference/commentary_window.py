"""Commentary Display Window with HLC Control"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
from datetime import datetime


class CommentaryWindow:
    """Displays model commentary and HLC control in a split-panel GUI."""

    def __init__(self, model_wrapper=None, config=None):
        self.window = None
        self.text_widget = None
        self.waypoint_display = None
        self.running = False
        self.thread = None
        self.message_queue = queue.Queue()
        self.last_message = None
        self.model_wrapper = model_wrapper
        self.config = config

        # Mode selection variables
        self.mode_var = None
        self.safety_var = None
        self.safety_frame = None
        self.qa_entry = None
        self.dreamer_entry = None
        self.target_speed = 0.0
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()
        import time
        time.sleep(0.5)
        
    def _run_window(self):
        self.window = tk.Tk()
        self.window.title("SimLingo AI")
        self.window.geometry("1400x600")  # Optimized for wide/short layout (bottom half of screen)
        self.window.configure(bg='#0a0e27')

        # Header - Compact
        header = tk.Frame(self.window, bg='#1a1f3a', height=45)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Title and nav mode indicator
        header_container = tk.Frame(header, bg='#1a1f3a')
        header_container.pack(expand=True, fill=tk.BOTH)

        tk.Label(header_container, text="SimLingo AI - Commentary & Control",
                font=("Segoe UI", 14, "bold"), bg='#1a1f3a', fg='#00d4ff').pack(side=tk.LEFT, padx=15, pady=10)

        # Nav mode indicator (set by model wrapper)
        self.nav_mode_label = tk.Label(header_container, text="",
                                       font=("Segoe UI", 9), bg='#1a1f3a', fg='#7a8fb5')
        self.nav_mode_label.pack(side=tk.RIGHT, padx=15, pady=10)

        # Mode Selection Panel - Compact horizontal layout
        mode_panel = tk.Frame(self.window, bg='#1a1f3a', height=60)
        mode_panel.pack(fill=tk.X, padx=10, pady=(0, 5))
        mode_panel.pack_propagate(False)

        # Single row container for all controls
        controls_row = tk.Frame(mode_panel, bg='#1a1f3a')
        controls_row.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left side: Task mode selection
        mode_container = tk.Frame(controls_row, bg='#1a1f3a')
        mode_container.pack(side=tk.LEFT)

        tk.Label(mode_container, text="Task:", font=("Segoe UI", 10, "bold"), bg='#1a1f3a', fg='#ff9500').pack(side=tk.LEFT, padx=(0, 10))

        # Radio buttons for mode selection
        self.mode_var = tk.StringVar(value='commentary')

        # Commentary mode
        tk.Radiobutton(mode_container, text="Commentary", variable=self.mode_var, value='commentary',
                      command=self._on_mode_change, bg='#1a1f3a', fg='#00d4ff', selectcolor='#0a0e27',
                      font=("Segoe UI", 10), activebackground='#1a1f3a', activeforeground='#00ff00').pack(side=tk.LEFT, padx=(0, 15))

        # Q&A mode
        tk.Radiobutton(mode_container, text="Q&A", variable=self.mode_var, value='qa',
                      command=self._on_mode_change, bg='#1a1f3a', fg='#00d4ff', selectcolor='#0a0e27',
                      font=("Segoe UI", 10), activebackground='#1a1f3a', activeforeground='#00ff00').pack(side=tk.LEFT, padx=(0, 15))

        # Dreamer mode
        tk.Radiobutton(mode_container, text="Dreamer", variable=self.mode_var, value='dreamer',
                      command=self._on_mode_change, bg='#1a1f3a', fg='#00d4ff', selectcolor='#0a0e27',
                      font=("Segoe UI", 10), activebackground='#1a1f3a', activeforeground='#00ff00').pack(side=tk.LEFT, padx=(0, 15))

        # Safety toggle (only visible when Dreamer is selected, default OFF)
        self.safety_frame = tk.Frame(mode_container, bg='#1a1f3a')
        self.safety_var = tk.BooleanVar(value=False)
        self.safety_check = tk.Checkbutton(self.safety_frame, text="Safety", variable=self.safety_var,
                                          command=self._on_safety_change, bg='#1a1f3a', fg='#ff9500', selectcolor='#0a0e27',
                                          font=("Segoe UI", 9), activebackground='#1a1f3a', activeforeground='#00ff00')
        self.safety_check.pack(side=tk.LEFT)
        # Initially hidden - will be shown when Dreamer mode is selected

        # Right side: Input fields (shown based on mode)
        self.input_container = tk.Frame(controls_row, bg='#1a1f3a')
        self.input_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(30, 0))

        # Q&A input
        self.qa_input_frame = tk.Frame(self.input_container, bg='#1a1f3a')
        tk.Label(self.qa_input_frame, text="Question:", font=("Segoe UI", 9), bg='#1a1f3a', fg='#7a8fb5').pack(side=tk.LEFT, padx=(0, 8))
        self.qa_entry = tk.Entry(self.qa_input_frame, font=("Consolas", 9), bg='#0f1419', fg='#e6f1ff', insertbackground='#00d4ff', width=50)
        self.qa_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.qa_entry.bind('<Return>', self._on_qa_submit)

        # Dreamer input
        self.dreamer_input_frame = tk.Frame(self.input_container, bg='#1a1f3a')
        tk.Label(self.dreamer_input_frame, text="Instruction:", font=("Segoe UI", 9), bg='#1a1f3a', fg='#7a8fb5').pack(side=tk.LEFT, padx=(0, 8))
        self.dreamer_entry = tk.Entry(self.dreamer_input_frame, font=("Consolas", 9), bg='#0f1419', fg='#e6f1ff', insertbackground='#00d4ff', width=50)
        self.dreamer_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.dreamer_entry.bind('<Return>', self._on_dreamer_submit)

        # Content
        content = tk.Frame(self.window, bg='#0a0e27')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # LEFT: Commentary
        left = tk.Frame(content, bg='#0a0e27')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(left, text="Model Commentary", font=("Segoe UI", 14, "bold"), bg='#1a1f3a', fg='#00d4ff', pady=10).pack(fill=tk.X)
        
        self.text_widget = scrolledtext.ScrolledText(left, wrap=tk.WORD, width=50, height=30, font=("Consolas", 13), bg='#0f1419', fg='#e6f1ff', padx=15, pady=15, state='disabled')
        self.text_widget.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        tk.Button(left, text="Clear History", command=self._clear_text, bg='#1e3a5f', fg='#ffffff', font=("Segoe UI", 10, "bold"), padx=20, pady=8).pack(pady=(0, 10))
        
        # RIGHT: Vehicle Status & Waypoints
        right = tk.Frame(content, bg='#0a0e27')
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(right, text="Vehicle Status", font=("Segoe UI", 14, "bold"), bg='#1a1f3a', fg='#ff9500', pady=10).pack(fill=tk.X)

        # Speed display - Current and Target side by side
        speed_container = tk.Frame(right, bg='#0f1419', padx=15, pady=15)
        speed_container.pack(fill=tk.X, pady=(5, 10))

        # Left: Current Speed
        current_speed_frame = tk.Frame(speed_container, bg='#0f1419')
        current_speed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(current_speed_frame, text="Current Speed", font=("Segoe UI", 11, "bold"), bg='#0f1419', fg='#ff9500').pack(anchor=tk.W, pady=(0, 5))
        self.current_speed_display = tk.Label(current_speed_frame, text="0.00 m/s", font=("Consolas", 16, "bold"), bg='#1a1f3a', fg='#00ff00', padx=15, pady=15, anchor=tk.CENTER)
        self.current_speed_display.pack(fill=tk.X)

        # Right: Target Speed
        target_speed_frame = tk.Frame(speed_container, bg='#0f1419')
        target_speed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(target_speed_frame, text="Target Speed", font=("Segoe UI", 11, "bold"), bg='#0f1419', fg='#ff9500').pack(anchor=tk.W, pady=(0, 5))
        self.target_speed_display = tk.Label(target_speed_frame, text="0.00 m/s", font=("Consolas", 16, "bold"), bg='#1a1f3a', fg='#00d4ff', padx=15, pady=15, anchor=tk.CENTER)
        self.target_speed_display.pack(fill=tk.X)

        # Waypoint display section (below HLC controls)
        waypoint_section = tk.Frame(right, bg='#0a0e27')
        waypoint_section.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        tk.Label(waypoint_section, text="Model Waypoints", font=("Segoe UI", 14, "bold"), bg='#1a1f3a', fg='#00d4ff', pady=10).pack(fill=tk.X)

        waypoint_container = tk.Frame(waypoint_section, bg='#0f1419', padx=15, pady=15)
        waypoint_container.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        # Waypoint text display
        self.waypoint_display = tk.Text(waypoint_container, wrap=tk.WORD, height=10, font=("Consolas", 10),
                                        bg='#0f1419', fg='#e6f1ff', padx=10, pady=10, state='disabled',
                                        borderwidth=0, highlightthickness=0)
        self.waypoint_display.pack(fill=tk.BOTH, expand=True)

        # Configure tags for waypoint display
        self.waypoint_display.tag_config('header', foreground='#ff9500', font=("Consolas", 10, "bold"))
        self.waypoint_display.tag_config('waypoint', foreground='#00d4ff', font=("Consolas", 10))
        self.waypoint_display.tag_config('speed', foreground='#00ff00', font=("Consolas", 11, "bold"))
        self.waypoint_display.tag_config('separator', foreground='#7a8fb5', font=("Consolas", 10))

        self._append_text("Waiting for AI commentary...")

        # Set nav mode indicator if model wrapper is available
        if self.model_wrapper and hasattr(self.model_wrapper, 'nav_mode'):
            nav_mode_text = "Target Point" if self.model_wrapper.nav_mode == 'target_point' else "HLC Command"
            self.nav_mode_label.config(
                text=f"Nav Mode: {nav_mode_text}",
                fg='#00ff00' if self.model_wrapper.nav_mode == 'target_point' else '#ff9500'
            )

        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._process_queue()
        self.window.mainloop()
        
    def _on_close(self):
        self.running = False
        if self.window:
            self.window.destroy()
            
    def _clear_text(self):
        if self.text_widget:
            self.text_widget.configure(state='normal')
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.configure(state='disabled')
    
    def _on_mode_change(self):
        """Handle mode selection change."""
        mode = self.mode_var.get()

        # Hide all input frames and safety toggle
        self.qa_input_frame.pack_forget()
        self.dreamer_input_frame.pack_forget()
        if self.safety_frame:
            self.safety_frame.pack_forget()

        # Show appropriate input frame
        if mode == 'qa':
            self.qa_input_frame.pack(fill=tk.BOTH, expand=True)
            self.qa_entry.focus()
            # Don't update model until user enters a question
            self.message_queue.put("[MODE] Q&A - Please enter a question and press Enter")
        elif mode == 'dreamer':
            # Show safety toggle next to Dreamer radio button
            if self.safety_frame:
                self.safety_frame.pack(side=tk.LEFT)
            self.dreamer_input_frame.pack(fill=tk.BOTH, expand=True)
            self.dreamer_entry.focus()
            # Don't update model until user enters an instruction
            self.message_queue.put("[MODE] Dreamer - Please enter an instruction and press Enter (Safety is OFF by default)")
        else:
            # Commentary mode - update immediately
            self._update_model_task_type()

    def _on_safety_change(self):
        """Handle safety toggle change."""
        # Only update if we're in Dreamer mode and have an instruction
        if self.mode_var.get() == 'dreamer':
            instruction = self.dreamer_entry.get().strip() if self.dreamer_entry else None
            if instruction:
                self._update_model_task_type()
                safety_str = "ON (reject unsafe instructions)" if self.safety_var.get() else "OFF (follow all instructions)"
                self.message_queue.put(f"[SAFETY] {safety_str}")

    def _on_qa_submit(self, event=None):
        """Handle Q&A question submission."""
        if self.qa_entry:
            question = self.qa_entry.get().strip()
            if question:
                self._update_model_task_type()
                self.message_queue.put(f"[Q&A] Question: {question}")

    def _on_dreamer_submit(self, event=None):
        """Handle Dreamer instruction submission."""
        if self.dreamer_entry:
            instruction = self.dreamer_entry.get().strip()
            if instruction:
                self._update_model_task_type()
                safety_str = "ON (reject unsafe instructions)" if self.safety_var.get() else "OFF (follow all instructions)"
                self.message_queue.put(f"[DREAMER] Instruction: {instruction} | Safety: {safety_str}")

    def _update_model_task_type(self):
        """Update the model wrapper with current task type settings."""
        if not self.model_wrapper:
            return

        mode = self.mode_var.get()

        if mode == 'commentary':
            self.model_wrapper.set_task_type('commentary')
            self.message_queue.put("[MODE] Commentary (Chain-of-Thought)")

        elif mode == 'qa':
            question = self.qa_entry.get().strip() if self.qa_entry else None
            if not question:
                # Don't update model if no question is entered
                self.message_queue.put("[WARNING] Please enter a question first")
                return
            self.model_wrapper.set_task_type('qa', question=question)

        elif mode == 'dreamer':
            instruction = self.dreamer_entry.get().strip() if self.dreamer_entry else None
            if not instruction:
                # Don't update model if no instruction is entered
                self.message_queue.put("[WARNING] Please enter an instruction first")
                return
            safety_enabled = self.safety_var.get()
            self.model_wrapper.set_task_type('dreamer', instruction=instruction, safety_enabled=safety_enabled)


    

    def _append_text(self, text):
        if self.text_widget and self.running:
            try:
                if text == self.last_message:
                    return
                self.last_message = text
                
                self.text_widget.configure(state='normal')
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.text_widget.insert(tk.END, "-" * 100 + "\n", 'separator')
                self.text_widget.insert(tk.END, f"[{timestamp}]\n", 'timestamp')
                clean_text = text.replace(" Waypoints:", "").strip()
                self.text_widget.insert(tk.END, f"{clean_text}\n\n", 'commentary')
                
                self.text_widget.tag_config('separator', foreground='#2d4f67', font=("Consolas", 10))
                self.text_widget.tag_config('timestamp', foreground='#7a8fb5', font=("Consolas", 11, "italic"))
                self.text_widget.tag_config('commentary', foreground='#00d4ff', font=("Consolas", 14, "bold"))
                
                self.text_widget.see(tk.END)
                self.text_widget.configure(state='disabled')
            except:
                pass
                
    def _process_queue(self):
        try:
            while True:
                text = self.message_queue.get_nowait()
                self._append_text(text)
        except queue.Empty:
            pass
        if self.running and self.window:
            self.window.after(100, self._process_queue)

    def update_commentary(self, text):
        """Add commentary text to the display queue (thread-safe)."""
        if text and text.strip():
            self.message_queue.put(text.strip())

    def update_speed(self, speed):
        """Update the current speed display (thread-safe)."""
        if hasattr(self, 'current_speed_display') and self.current_speed_display:
            try:
                self.current_speed_display.config(text=f"{speed:.2f} m/s")
            except:
                pass

    def update_waypoints(self, route_waypoints, speed_waypoints):
        """
        Update the waypoint display with model predictions (thread-safe).

        Args:
            route_waypoints: Route waypoints array [F, 2] in ego frame
            speed_waypoints: Speed waypoints array [F, 2] in ego frame
        """
        if not hasattr(self, 'waypoint_display') or self.waypoint_display is None:
            return

        try:
            import numpy as np

            # Calculate target speed using same method as controller
            # Reference: simlingo/team_code/agent_simlingo.py control_pid() method
            # Uses waypoints[0] to waypoints[2] (first 0.5s of predictions)
            if speed_waypoints is not None and len(speed_waypoints) > 2:
                # Calculate model's predicted speed (in CARLA scale)
                model_target_speed = np.linalg.norm(speed_waypoints[2] - speed_waypoints[0]) * 2.0

                # Scale to QCar2 range for display (same as control_converter.py)
                if self.config and hasattr(self.config, 'speed_scale'):
                    target_speed = model_target_speed * self.config.speed_scale
                else:
                    # Fallback if config not available
                    target_speed = model_target_speed
            else:
                target_speed = 0.0

            # Update target speed display
            self.target_speed = target_speed
            if hasattr(self, 'target_speed_display') and self.target_speed_display:
                try:
                    self.target_speed_display.config(text=f"{target_speed:.2f} m/s")
                except:
                    pass

            # Format waypoint text
            # SimLingo outputs: 10 speed waypoints, 20 route waypoints
            waypoint_text = ""

            # Target speed section
            waypoint_text += "Target Speed:\n"
            waypoint_text += f"  {target_speed:.2f} m/s\n\n"

            # Prepare route and speed waypoint data
            # Show first 5 waypoints for each to keep display compact
            num_to_show = 5
            route_lines = []
            speed_lines = []

            # Route waypoints section (geometric path) - Model outputs 20 waypoints
            if route_waypoints is not None and len(route_waypoints) > 0:
                for i in range(min(num_to_show, len(route_waypoints))):
                    x, y = route_waypoints[i]
                    route_lines.append(f"[{i}] x:{x:6.2f}m y:{y:6.2f}m")

                if len(route_waypoints) > num_to_show:
                    route_lines.append(f"... ({len(route_waypoints) - num_to_show} more)")

            # Speed waypoints section (velocity-based path) - Model outputs 10 waypoints
            if speed_waypoints is not None and len(speed_waypoints) > 0:
                for i in range(min(num_to_show, len(speed_waypoints))):
                    x, y = speed_waypoints[i]
                    speed_lines.append(f"[{i}] x:{x:6.2f}m y:{y:6.2f}m")

                if len(speed_waypoints) > num_to_show:
                    speed_lines.append(f"... ({len(speed_waypoints) - num_to_show} more)")

            # Create side-by-side display
            waypoint_text += "Route (geometric)          Speed (velocity)\n"
            waypoint_text += "─" * 27 + "│" + "─" * 27 + "\n"

            max_lines = max(len(route_lines), len(speed_lines))
            for i in range(max_lines):
                route_part = route_lines[i] if i < len(route_lines) else ""
                speed_part = speed_lines[i] if i < len(speed_lines) else ""
                waypoint_text += f"{route_part:<27}│ {speed_part}\n"

            # Update display
            self.waypoint_display.configure(state='normal')
            self.waypoint_display.delete(1.0, tk.END)

            # Insert with formatting
            lines = waypoint_text.split('\n')
            for line in lines:
                if 'Target Speed:' in line:
                    self.waypoint_display.insert(tk.END, line + '\n', 'header')
                elif 'Route (geometric)' in line and 'Speed (velocity)' in line:
                    # Header line for side-by-side display
                    self.waypoint_display.insert(tk.END, line + '\n', 'header')
                elif '─' in line or '│' in line:
                    # Separator line
                    self.waypoint_display.insert(tk.END, line + '\n', 'separator')
                elif 'm/s' in line and 'Target Speed' not in line:
                    self.waypoint_display.insert(tk.END, line + '\n', 'speed')
                elif line.strip().startswith('[') or '│' in line:
                    self.waypoint_display.insert(tk.END, line + '\n', 'waypoint')
                else:
                    self.waypoint_display.insert(tk.END, line + '\n')

            self.waypoint_display.configure(state='disabled')

        except Exception:
            pass

    def update_commentary(self, text):
        if not self.running:
            return
        if not text or text.strip() == "":
            text = "(empty)"
        try:
            self.message_queue.put(text)
        except Exception:
            pass

    def stop(self):
        self.running = False
        if self.window:
            try:
                self.window.quit()
            except:
                pass
