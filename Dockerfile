FROM python:3.6.9

EXPOSE 8501

WORKDIR /app

COPY requirement.txt ./requirement.txt

RUN pip3 install -r requirement.txt

COPY . .

CMD streamlit run app.py
