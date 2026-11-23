def save_log(diccionario,file_name):
    with open(file_name,"w",encoding="utf-8") as file:
        for metrica in diccionario:
            file.write(f"{metrica}: {diccionario[metrica]}\n")

