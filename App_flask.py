from flask import Flask,redirect,url_for,render_template,request

app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('appindex.html')

@app.route('/prediction',methods=['POST','GET'])
def calc():
    import pandas as pd
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    from matplotlib import pyplot as plt
    data=pd.read_excel("DUET Data - Copy.xlsx")
    data=data.drop(['Unnamed: 0'], axis=1)
    data=data.dropna()
    data=data[data['Graduation Marks']>=50]
    x=data[['Graduation Marks','Entrance Marks']]
    x_scaled=preprocessing.scale(x)
    kmeans_new=KMeans(4)
    kmeans_new.fit(x_scaled)
    clusters_new=x.copy()
    clusters_new['clusters']=kmeans_new.fit_predict(x_scaled)
    sns.scatterplot(x='Entrance Marks',y='Graduation Marks',hue='clusters',data=clusters_new,palette='Paired')
    plt.savefig('static/myplot.png')
    input_para=[float(request.form['Graduation Marks']),float(request.form['Entrance Marks'])]

    input_para[1]=(input_para[1]-data['Entrance Marks'].mean())/data['Entrance Marks'].std()
    input_para[0]=(input_para[0]-data['Graduation Marks'].mean())/data['Graduation Marks'].std()
    print(input_para)
    print(kmeans_new.predict([input_para]))

    return render_template('appresult.html',result=kmeans_new.predict([input_para]),plot_url='static/myplot.png')
    

app.run()
