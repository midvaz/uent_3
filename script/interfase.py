
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog



from network_unet import image_processing
from network_unet import train


class Interfase:
    def __init__(self):
        """
        Магический метод, необходимый для 
        инициализации неоходимых переменных
        """
        pass

    def __select_file(self)->None:
        """
        Обозначает в выборе файлов расширения, 
        которые можно загрузить в программу
        """
        filetypes = (
            ("JPEG files", "*.JPG"), ("JPEG files", "*.jpg"), 
            ("PNG files" , "*.png"), ("All files", "*.*")
        )
        filepath = fd.askopenfilename(title="Выберите файл", initialdir="/", filetypes=filetypes)
        self.label_file.config(text=filepath)


    def __show_image(self)->None:
        """
        Отображение обработанной фотографии
        """
        # TODO:обрабатывать полученную переменную
        _ = self.__select()
        self.canvas.delete("all")
        self.filepath = self.label_file.cget("text")
        img = ImageTk.PhotoImage(Image.open(f"{self.filepath}").resize((600,600), Image.ANTIALIAS))
        self.canvas.create_image(400, 100, image=img)
        self.canvas.create_image(512, 512, image=img)
        self.canvas.image = img

        self.label_file = Label(self.tk, text=analis, width=60)
        self.label_file.grid(row= 5, column=2, padx=10, pady =10)


    def __on_closing(self)->None:
        """
        Метод, который вызывается при закрытие приложения
        """
        if messagebox.askokcancel("Выход из приложения", "Хотите выйти из приложения?"):
            self.tk.destroy()


    def __select(self)->dict:
        """
        Метод который обрабатывает то, что выполняет функция
        """
        markers = {
            "roud":False,
            "forest":False,
            "field":False,
            "hole":False
        }
        # if self.roud.get() == 1: markers["roud"] = True
        # if self.forest.get() == 1:  markers["forest"] = True
        # if self.field.get() == 1:  markers["field"] = True
        # if self.hole.get() == 1:  markers["hole"] = True 
        # print(markers)
        return markers
    

    def __get_analisized(self):
        """
        Функция по получению анлиза файла
        """
        arial_list = image_processing.img_analysis(img_path=self.filepath)
        text = ""
        if arial_list[0] > 0:
            text += f"Площадь поля={arial_list[0]} м2\n"
        if arial_list[1] > 0:
            text += f"Площадь ям={arial_list[1]} м2\n"
        if arial_list[2] > 0:
            text += f"Площадь леса={arial_list[2]} м2\n"
        if arial_list[3] > 0:
            text += f"Площадь поля={arial_list[3]} м2"
        
        if (arial_list[1] > 0) and (arial_list[0] > 0):
            text += f"Процент проблемных зон на поле равен:\n{image_processing.get_percent(arial_list[1], 92160)}%"

        return text
        

    def __check_correctness_input_rgb(self, input_text:str)->bool:
        """
        Проверка корректности ввода rgb
        """
        list_input = input_text.split(',')
        if len(list_input) != 3:
            self.errmsg.set("Должно быть 3-и числа, \nразделенные запятой без букв и \nдругих знаков припинания")
            return False
        for i in list_input:
            if (not i.isdigit()) or (int(i) < 0) or (int(i) > 255):
                self.errmsg.set("Должно быть 3-и числа, \nразделенные запятой без букв и \nдругих знаков припинания")
                return False
        self.errmsg.set("")
        return True


    def __start_train_with_new_data(self):
        """
        Функция начала обучения
        """
        if (self.new_class_name!="")and(self.rgb!="")and(self.new_img_path!="")and(self.new_masks_path!=""):
            self.errmsg.set("")
            self.window.destroy()
            cnn = train.Unet(
                new_rgb=self.rgb,
                new_class_name=self.new_class_name,
                path_img=self.new_img_path,
                path_mask=self.new_masks_path
            )
            cnn.make_dataset()
            cnn.creating_model()
            cnn.build_network()
            cnn.train_network()
            cnn.save_model()
        else:
            self.errmsg.set("Необходимо ввести все данные.")


    def __new_window(self):
        """
        Создание нового окна
        """
        self.window = Toplevel()
        self.window.title("Добавление нового класса")
        self.window.geometry("400x400")


    def __add_new_class(self):
        """
        Добавления нового класса
        """
        self.__new_window()

        self.new_class_name = ""
        self.rgb = ""

        self.new_img_path = ""
        self.new_masks_path = ""

        def __show_message():
            if (self.new_class_name == "") and (self.rgb == ""):
                self.new_class_name = entry1.get() 
                self.rgb = entry2.get() 
            if self.__check_correctness_input_rgb(self.rgb):
               self.new_img_path = __callback()
        
        def __show_message_masks():
            if (self.new_class_name == "") and (self.rgb == ""):
                self.new_class_name = entry1.get() 
                self.rgb = entry2.get() 
            if self.__check_correctness_input_rgb(self.rgb):
               self.new_masks_path = __callback()
            
        def __callback():
            label3["text"] = ""
            return  filedialog.askdirectory()
            
        
        entry1 = ttk.Entry(self.window)
        entry1.grid(row= 1, column=1, padx=10, pady =10)

        entry2 = ttk.Entry(self.window)
        entry2.grid(row= 2, column=1, padx=10, pady =10)
        
        btn1 = ttk.Button(self.window, text="Выбрать папку с \nобучающими данными", command=__show_message)
        btn1.grid(row= 10, column=1, padx=10, pady =10)

        btn2 = ttk.Button(self.window, text="Выбрать папку с \nобущающими масками", command=__show_message_masks)
        btn2.grid(row= 11, column=1, padx=10, pady =10)

        label1 = ttk.Label(self.window, text="Введите имя нового класса")
        label1.grid(row= 1, column=2, padx=10, pady =10)

        label2= ttk.Label(self.window, text="Введите RGB цвета нового класса\nПример:255, 255, 255")
        label2.grid(row= 2, column=2, padx=10, pady =10)

        self.errmsg = StringVar()

        label3= ttk.Label(self.window, foreground="red", textvariable=self.errmsg)
        label3.grid(row= 4, column=2, padx=10, pady =10)

        btn3 = ttk.Button(self.window, text="Начать обучение", command=self.__start_train_with_new_data)
        btn3.grid(row= 13, column=1, padx=10, pady =20)

        self.window.grab_set()


    def rendering(self)->None:
        """
        Отрисовка всех элементов стартового экрана
        """
        self.tk = Tk()
    
        self.style = ttk.Style()
        self.style.theme_use("xpnative")

        self.tk.protocol("WM_DELETE_WINDOW", self.__on_closing)
        self.tk.title("Приложение")
        self.tk.resizable(0, 0)
        self.tk.wm_attributes("-topmost", 1)

        self.tk.title("Выбор фотографии")

        self.label_file = Label(self.tk, text="Файл не выбран", width=60)
        self.label_file.grid(row= 1, column=2, padx=10, pady =10)

        self.btn_select = Button(self.tk, text="Выбрать", command=self.__select_file)
        self.btn_select.grid(row= 2, column=2, padx=10, pady =10)

        self.btn_show = Button(self.tk, text="Показать", command=self.__show_image)
        self.btn_show.grid(row= 3, column=2, padx=10, pady =10)

        self.canvas = Canvas(self.tk, width=600, height=600, bd=0, highlightthickness=0)
        self.canvas.grid(row= 5, column=1, padx=10, pady =10)

        self.btn_show = Button(self.tk, text="Добавление нового класса", command=self.__add_new_class)
        self.btn_show.grid(row= 3, column=1, padx=10, pady =10)
        
        self.tk.mainloop()

if __name__ == "__main__":
    interfase = Interfase()
    interfase.rendering()