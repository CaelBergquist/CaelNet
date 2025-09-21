from model import SimpleNN
import os

#model = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)

#print(model.get_inner_probability())



def prepend_folder(folder, prepend_str):
    for filename in os.listdir(folder):
        new_filename = prepend_str + filename
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))

def delete_first_char(folder):
    for filename in os.listdir(folder):
        new_filename = filename[1:]  # Remove the first character
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))

prepend_folder("images2", "8")
#delete_first_char("images2")
