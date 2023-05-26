
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image



def select_file():
    filetypes = (("JPEG files", "*.JPG"), ("JPEG files", "*.jpg"), ("PNG files" , "*.png"), ("All files", "*.*"))
    filepath = fd.askopenfilename(title="Выберите файл", initialdir="/", filetypes=filetypes)
    label_file.config(text=filepath)


def show_image():
    select()
    canvas.delete("all")
    filepath = label_file.cget("text")
    img = ImageTk.PhotoImage(Image.open(f"{filepath}"))
    canvas.create_image(400, 100, image=img)
    canvas.image = img


def on_closing():
    if messagebox.askokcancel("Выход из приложения", "Хотите выйти из приложения?"):
        tk.destroy()


def select():
    markers = {
        "roud":False,
        "forest":False,
        "field":False,
        "hole":False
    }
    if roud.get() == 1: markers["roud"] = True
    if forest.get() == 1:  markers["forest"] = True
    if field.get() == 1:  markers["field"] = True
    if hole.get() == 1:  markers["hole"] = True 
    print(markers)


if __name__ == "__main__":
    tk = Tk()
  
    style = ttk.Style()
    style.theme_use("xpnative")

    tk.protocol("WM_DELETE_WINDOW", on_closing)
    tk.title("Приложение")
    tk.resizable(0, 0)
    tk.wm_attributes("-topmost", 1)

    tk.title("Выбор фотографии")

    position = {"padx":6, "pady":6, "anchor":NW}

    label_file = Label(tk, text="Файл не выбран", width=60)
    # label_file.pack(pady=10)
    label_file.grid(row= 1, column=2, padx=10, pady =10)

    btn_select = Button(tk, text="Выбрать", command=select_file)
    btn_select.grid(row= 2, column=2, padx=10, pady =10)

    btn_show = Button(tk, text="Показать", command=show_image)
    btn_show.grid(row= 3, column=2, padx=10, pady =10)

    roud = IntVar()
    roud_checkbutton = ttk.Checkbutton(text="roud", variable=roud)
    # roud_checkbutton.pack(**position)
    roud_checkbutton.grid(row= 1, column=1, padx=10, pady =10)

    forest = IntVar()
    forest_checkbutton = ttk.Checkbutton(text="forest", variable=forest)
    forest_checkbutton.grid(row= 2, column=1, padx=10, pady =10)

    field = IntVar()
    field_checkbutton = ttk.Checkbutton(text="field", variable=field)
    field_checkbutton.grid(row= 3, column=1, padx=10, pady =10)

    hole = IntVar()
    hole_checkbutton = ttk.Checkbutton(text="hole", variable=hole)
    hole_checkbutton.grid(row= 4, column=1, padx=10, pady =10)

    canvas = Canvas(tk, width=600, height=400, bd=0, highlightthickness=0)

    canvas.grid(row= 5, column=1, padx=10, pady =10)

    tk.mainloop()