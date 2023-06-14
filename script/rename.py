import os 


def rename_files_in_directory(
        list_dir:list = ['./img', './masks_machine', "./masks_human"]
    )->None:
    """
    Переименновка файлов для корректной загрузки датасета
    """
    for i in range(len(list_dir)):
        for temp, f in enumerate(sorted(os.listdir(f"{list_dir[i]}"))):
            os.rename(f"{list_dir[i]}/{f}", f"{list_dir[i]}/{temp}.{f[-3:]}")


