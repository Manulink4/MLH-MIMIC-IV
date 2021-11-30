import os
import sys

def setabspath(**kwargs):
            filePath = __file__
            absFilePath = os.path.abspath(__file__)
            dname = os.path.dirname(absFilePath)
            os.chdir(dname)
            # print(dname)
            print(kwargs)
            if kwargs == {}:
                print(dname)
                os.system('call C:/Users/Pablo/anaconda3/Scripts/activate.bat C:/Users/Pablo/anaconda3 && conda activate st && streamlit run inputfile.py')
                    # os.system('call C:/Users/Pablo/anaconda3/Scripts/activate.bat C:/Users/Pablo/anaconda3 && conda activate st && streamlit run inputfile.py')
            else:
                path = kwargs["path"].strip()
                base = "/".join(i for i in kwargs["path"].strip().split("/")[:4])
                env  = kwargs["env"] 
                os.system('call {} {} && conda activate {} && streamlit run inputfile.py'.format(path,base,env))
                
if __name__=="__main__":
    # admin = None
    admin = 0
    c = 0
    while admin not in [0,1]:
        if c>0:
            print("Please enter 0 if you are the adminer and 1 otherwise \n")
        admin = int(input("""
                    Adminer of the content ?:
                    0. Yes
                    1. No 
                    >"""))
        c+=1
    if admin ==0:
        setabspath()      
    else:
        path = str(input("Enter the path were activte.bat from conda is located: "))
        env  = str(input("Enter the conda environment were yoyu have installed your dependencies"))
        setabspath(path=path,env=env)

# call <anaconda_dir>/Scripts/activate.bat <anaconda_dir>
