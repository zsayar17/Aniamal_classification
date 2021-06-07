import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer




verisayisi=100
pikseller=[]
hayvanlar=[]
translate = {"cane": "Köpek", "cavallo": "At", "elefante": "Fil", "farfalla": "Kelebek", "gallina": "Tavuk", "gatto": "Kedi", "mucca": "İnek", "pecora": "Koyun", "scoiattolo": "Sincap", "ragno": "Örümcek"}



### DOSYA OKUMA
pathh=r'C:\Users\ahmet\Desktop\pyilederinogrenme\raw-img'
foldernames = os.listdir(pathh)



for kategori in translate:    
    path=os.path.join(pathh,kategori)
    q=os.listdir(path)
    b=0
    for img in os.listdir(path):
        
        img_array=cv2.imread(os.path.join(path,img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array=cv2.resize(img_array,(224,224))
        pikseller.append(img_array)
        hayvanlar.append(translate[kategori])
        b+=1
        if b==verisayisi:
            break


 

#Gradyan inişlerinin Daha iyi olabilmesi için normalize ettik
def normalize(piksel):
    piksel=np.array(piksel)

    piksel=piksel/255  
    return piksel




#hayvanlar kümesini kategorileştirdik
def kategorileştir(hayvanlar):
    encoder = LabelBinarizer()
    hayvanlar2=np.array(hayvanlar)

    hayvanlar2 = encoder.fit_transform(hayvanlar2)
    
    return hayvanlar2

#encoderda karmaşık gelen verierli düzelltik
def düzelt(hayvanlar,verisayisi):
    genislik=(len(hayvanlar))
    genislik/=verisayisi #  kaç tane hayvanımız varsa onu aldık
    liste=[] # verisetlerini farklı listelere ayırıp bunun iöine atadık
    liste2=[] # düzeltilmiş halini buraya aldık
    sayi=verisayisi # aralık için aldık
    sayi2=0 #aralık için aldık
    sayi3=0 #10 tane kategorimizi 0 dan başlayıp arttırabilmek için tanımladık
    for i in range(int(genislik)):
        liste.append(hayvanlar[sayi2:sayi])
        sayi+=verisayisi
        sayi2+=verisayisi
        
        
    for j in range(int(genislik)): 
        for i in liste:
           
            if i[0][sayi3]==1:
                
                liste2.append(i)
                sayi3+=1
                break
    
    
    
    liste2=np.array(liste2)
    liste2=liste2.reshape(len(hayvanlar),int(genislik))
    
    return liste2



        
hayvanlar2=kategorileştir(hayvanlar)   
pikseller=normalize(pikseller)   
hayvanlar3=düzelt(hayvanlar2,verisayisi)




#veri setini böldük

train_x, test_x, train_y, test_y = train_test_split(pikseller, hayvanlar3
                                        , test_size=0.20)

"""

### MODEL

giris_sekli=(224,224,3)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),input_shape = giris_sekli))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))


model.add(layers.Conv2D(32,(3,3)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))


model.add(layers.Conv2D(64,(3,3)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))


model.add(layers.Conv2D(64,(3,3)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))


model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5)) #0.5
model.add(layers.Dense(10))
model.add(layers.Activation("softmax"))
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])





### EĞİTİM


model.fit(
    
  train_x,
  train_y,
  epochs = 25)


val_loss,val_acc=model.evaluate(test_x,test_y) 
print(val_loss,val_acc)
model.save("dl_model_kayit3_8hayvan.h5")
#"""


### DENEME

model=load_model("model_kayit_hayvan.h5")
#model.summary()







#tahminler içinde en yükseğini bulduk ve kıyasladık


def kıyaslama(pred):
    
    
    for predd in pred:
        
        genislik=len(predd)   
        for i in range(genislik):
            b=0
            
            
            
            for j in predd:
                
                ilksayi=predd[i]
                ikincisayi=j
                if ilksayi>=ikincisayi:
                    
                
                    b+=1
                    
            
            
            #tahminin hangi hayvana geldiğini yazdırdık
            if b==genislik:
                z=0
                for q in translate:
                   
                   
                   if z==i:
                       tahmin=translate[q]
                   z+=1    
                break
    return tahmin



"""
#dışardan resim

img=cv2.imread("ayi.jpg")

img = cv2.resize(img,(224,224))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img=np.array(img)

img=normalize(img)

plt.title("resim"),plt.imshow(img)

img = img.reshape(1,224,224,3)

pred=model.predict(img)



print("\n\n\n\n\n\n")
print(kıyaslama(pred))




"""

#veriden aldık


sayi=901



plt.title(hayvanlar[sayi]),plt.imshow(pikseller[sayi])

plt.show()

img=pikseller[sayi].reshape(1, 224, 224, 3)

pred=model.predict(img)



print("\n\n\n\n\n\n")
print(kıyaslama(pred))


#"""
