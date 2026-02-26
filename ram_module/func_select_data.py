from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def select_site_data(
    allowed_exts={".csv", ".xlsx", ".xls"},
    title="Select a data file"
) -> Path:
    """
    Open a native file dialog (always on top of Spyder) and return a Path to a .csv/.xlsx/.xls file.
    Keeps prompting until a valid file is chosen or the user cancels.
    """
    root = tk.Tk()
    # Make the hidden root topmost so dialogs open in front of Spyder
    root.withdraw()
    root.update_idletasks()
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()

    try:
        while True:
            path_str = filedialog.askopenfilename(
                title=title,
                filetypes=[
                    ("Data files", "*.csv *.xlsx *.xls"),
                    ("CSV", "*.csv"),
                    ("Excel", "*.xlsx *.xls"),
                    ("All files", "*.*"),
                ],
                parent=root,  # ensure dialog is owned by our topmost root
            )
            if not path_str:
                # User cancelled: close the hidden root before exiting
                raise FileNotFoundError("No file selected.")

            p = Path(path_str)
            if p.suffix.lower() in allowed_exts:
                print(f"Selected: {p}")
                return p

            messagebox.showerror(
                "Invalid file",
                f"Please select one of: {', '.join(sorted(allowed_exts))}",
                parent=root,  # keep error on top too
            )
    finally:
        # Always destroy the root (even on errors) so no ghost window sticks around
        try:
            root.destroy()
        except Exception:
            pass
