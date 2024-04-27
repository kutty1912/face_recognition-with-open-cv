import cv2 as c
import face_recognition
kutty= c.imread("veena.jpeg")
rgb_kutty = c.cvtColor(kutty, c.COLOR_BGR2RGB)
encode_kutty = face_recognition.face_encodings(rgb_kutty)[0]

chinnu = c.imread("darshan.jpeg")
rgb_chinnu = c.cvtColor(chinnu, c.COLOR_BGR2RGB)
encode_chinnu = face_recognition.face_encodings(rgb_chinnu)[0]

wishi= c.imread("vaishnavi.jpeg")
rgb_wishi = c.cvtColor(wishi, c.COLOR_BGR2RGB)
encode_wishi= face_recognition.face_encodings(rgb_wishi)[0]

wis= c.imread("gisa mam.jpeg")
rgb_wis = c.cvtColor(wis, c.COLOR_BGR2RGB)
encode_wis= face_recognition.face_encodings(rgb_wis)[0]

sir= c.imread("jayavadivel.jpeg")
rgb_sir = c.cvtColor(sir, c.COLOR_BGR2RGB)
encode_sir= face_recognition.face_encodings(rgb_sir)[0]

mom= c.imread("sarada.jpeg")
rgb_mom = c.cvtColor(mom, c.COLOR_BGR2RGB)
encode_mom= face_recognition.face_encodings(rgb_mom)[0]

aachi= c.imread("sreelatha.jpeg")
rgb_aachi = c.cvtColor(aachi, c.COLOR_BGR2RGB)
encode_aachi= face_recognition.face_encodings(rgb_aachi)[0]

bro= c.imread("kshitij.jpeg")
rgb_bro = c.cvtColor(bro, c.COLOR_BGR2RGB)
encode_bro= face_recognition.face_encodings(rgb_bro)[0]

kar= c.imread("karun.jpeg")
rgb_kar = c.cvtColor(bro, c.COLOR_BGR2RGB)
encode_kar= face_recognition.face_encodings(rgb_kar)[0]

sir= c.imread("satish_sir.jpeg")
rgb_sir = c.cvtColor(bro, c.COLOR_BGR2RGB)
encode_sir= face_recognition.face_encodings(rgb_sir)[0]

result = face_recognition.compare_faces([encode_chinnu], encode_wis)
print("result",result)

c.imshow("veena",kutty)
c.imshow("darshan",chinnu)
c.imshow("vaishnavi",wishi)
c.imshow("gisa mam",wis)
c.imshow("jayavadivel",sir)
c.imshow("sarada",mom)
c.imshow("sreelatha",aachi)
c.waitKey(0)
