wget http://perso.esiee.fr/~dpt-it/a2si/heart/data/HeartDatabase.tgz
tar -zxvf HeartDatabase.tgz
rm HeartDatabase.tgz

find HeartDatabase/ -iname "out*.pgm" > patients.txt
find HeartDatabase/ -iname "*_scaled.pgm" > experts.txt

python3 main.py 2>/dev/null

