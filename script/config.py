import json

def get_configuration()->dict:
    """
    Возращает изначальные настройки нейронной сети
    """
    with open('config.json', 'r') as f:
        data = json.load(f)
    
    return data

if __name__ == '__main__':
  print(get_configuration())