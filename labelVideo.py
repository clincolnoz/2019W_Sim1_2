import time
import pandas as pd

# %% Inputs
input_var = "not_x"
df = pd.DataFrame(columns=['Label','Time'])
while input_var!= "x":
    input_var = input("Kermit (f/b = front/back, d/g = left/right, t = talking)? ")
    print(input_var)

    if input_var == "s":
        start = time.time()
    elif input_var == "e":
        end = time.time()
    elif input_var == "f":
        delta = time.time()-start
        df=df.append(pd.DataFrame({'Label': [input_var], 'Time': [delta]}),ignore_index=True)
     

print(df)


