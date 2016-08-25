# Python Environment
FROM python:3-onbuild

ADD http://www.pjreddie.com/media/files/mnist_train.csv mnist_train.csv

CMD [ "python", "simpleNN.py" ]
