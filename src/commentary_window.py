"""Commentary Display Window with HLC Control"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
from datetime import datetime


class CommentaryWindow:
    """Displays model commentary and HLC control in a split-panel GUI."""
    
    def __init__(self, model_wrapper=None):
        self.window = None
        self.text_widget = None
        self.hlc_entry = None
        self.current_hlc_display = None
        self.waypoint_display = None
        self.running = False
        self.thread = None
        self.message_queue = queue.Queue()
        self.last_message = None
        self.model_wrapper = model_wrapper
        
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
        self.window.geometry("1200x700")
        self.window.configure(bg='#0a0e27')
        
        # Header
        header = tk.Frame(self.window, bg='#1a1f3a', height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="SimLingo AI - Commentary & Control", font=("Segoe UI", 18, "bold"), bg='#1a1f3a', fg='#00d4ff').pack(pady=20)
        
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
        
        # RIGHT: HLC (split into two columns)
        right = tk.Frame(content, bg='#0a0e27')
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(right, text="High-Level Commands", font=("Segoe UI", 14, "bold"), bg='#1a1f3a', fg='#ff9500', pady=10).pack(fill=tk.X)

        # Container for side-by-side layout
        hlc_container = tk.Frame(right, bg='#0f1419')
        hlc_container.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        # LEFT: Current Speed
        left_hlc = tk.Frame(hlc_container, bg='#0f1419', padx=15, pady=20)
        left_hlc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Current speed display (prominent at top)
        tk.Label(left_hlc, text="Current Speed:", font=("Segoe UI", 14, "bold"), bg='#0f1419', fg='#ff9500').pack(anchor=tk.W, pady=(0, 10))
        self.current_speed_display = tk.Label(left_hlc, text="0.00 m/s", font=("Consolas", 20, "bold"), bg='#1a1f3a', fg='#00ff00', padx=20, pady=20, anchor=tk.CENTER)
        self.current_speed_display.pack(fill=tk.X)

        # RIGHT: Enter Command
        right_hlc = tk.Frame(hlc_container, bg='#0f1419', padx=15, pady=20)
        right_hlc.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right_hlc, text="Enter Command (Press Enter):", font=("Segoe UI", 11, "bold"), bg='#0f1419', fg='#ff9500').pack(anchor=tk.W, pady=(0, 10))
        self.hlc_entry = tk.Entry(right_hlc, font=("Consolas", 11), bg='#1a1f3a', fg='#e6f1ff', insertbackground='#00d4ff')
        self.hlc_entry.pack(fill=tk.X, pady=(0, 15), ipady=8)
        self.hlc_entry.bind('<Return>', self._on_hlc_submit)

        # Current command display (small, under text entry)
        tk.Label(right_hlc, text="Active:", font=("Segoe UI", 9), bg='#0f1419', fg='#7a8fb5').pack(anchor=tk.W, pady=(10, 5))
        self.current_hlc_display = tk.Label(right_hlc, text="[Default]", font=("Consolas", 9, "italic"), bg='#1a1f3a', fg='#7a8fb5', padx=10, pady=8, anchor=tk.W, wraplength=220, justify=tk.LEFT)
        self.current_hlc_display.pack(fill=tk.X)

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
    
    def _on_hlc_submit(self, event=None):
        if self.hlc_entry and self.model_wrapper:
            command = self.hlc_entry.get().strip()
            if command:
                # Set new command
                self.model_wrapper.set_hlc(command)
                self.message_queue.put(f"[HLC SET] {command}")
                if self.current_hlc_display:
                    self.current_hlc_display.config(text=command, fg='#00ff00')
                self.hlc_entry.delete(0, tk.END)
            else:
                # Empty entry = clear command
                self.model_wrapper.set_hlc(None)
                self.message_queue.put("[HLC CLEARED] Using default behavior")
                if self.current_hlc_display:
                    self.current_hlc_display.config(text="[Default]", fg='#7a8fb5')
    

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

            # Calculate target speed from last two speed waypoints
            # Waypoints are 0.25s apart, so multiply by 4 to get m/s
            if speed_waypoints is not None and len(speed_waypoints) >= 2:
                last_wp = speed_waypoints[-1]
                second_last_wp = speed_waypoints[-2]
                delta = last_wp - second_last_wp
                target_speed = np.linalg.norm(delta) * 4.0
            else:
                target_speed = 0.0

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
