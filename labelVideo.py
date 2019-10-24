import time
import pandas as pd
import csv

# %% Inputs
filename = input('filename: ')

input_var = "not_x"
# df = pd.DataFrame(columns=['Label','Time'])

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file, delimiter = ",")

    while input_var!= "x":
        input_var = input("Kermit (f = kermit, t = talking, g = no kermit)? ")
        print(input_var)

        if input_var == "s":
            start = time.time()
            writer.writerow(('s',0))

        elif input_var == "e":
            delta = time.time()-start
            writer.writerow(('e',delta))

        elif input_var == "f":
            delta = time.time()-start

            writer.writerow(('k',delta))

            # df=df.append(pd.DataFrame({'Label': [input_var], 'Time': [delta]}),ignore_index=True)

        elif input_var == "t":
            delta = time.time()-start

            writer.writerow(('kt',delta))

        elif input_var == "g":
            delta = time.time()-start

            writer.writerow(('nok',delta))


     




