#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# script to get molecular weights i


import numpy as np
import plotly   #may need to install this library...
import pandas as pd
import sklearn
import sys
sys.path.append('../import')
from feature_names import  transl_symbols
from drop_compounds import drop_compounds

import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#load and prepare the data
#change route to the place where the data lives
route="/home/lsimon/jobs/pka/Gpka/"

#data=pd.read_csv(route+"values_extracted-ext-pbeh3c.csv")
data=pd.read_csv(route+"extracted_data/values_extracted-gibbs-swb97xd.25.csv",low_memory=False,encoding='latin_1')
data.drop(["reference"],axis=1,inplace=True)
data.dropna(axis=0)
data.dropna()


data2=pd.read_csv(route+"extracted_data/values+alpha-beta-gamma-gibbs-swb97xd.csv",low_memory=False,encoding='latin_1')
data2.dropna(axis=0)
data2.dropna()
data2.drop(["pKa"],axis=1,inplace=True) 
data2.drop(["Unnamed: 0"],axis=1,inplace=True)
data=pd.merge(data,data2,on="compn")
data.info(verbose=True,show_counts=True)
for d in drop_compounds:    data =data[data["compn"].str.startswith(d)==False]

"""
#simple linear regression model... just to find out errors in pKa or missing conformations
#point colors reflects the charge of the protonated compound (0 in acetic acid, +1 in methylamine, etc.)
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

test_attributes=['deltaG']

X,Y=np.c_[data[test_attributes]],data["pKa"].copy()
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

pka_prediction=lin_reg.predict(X)

#data['colors'] = data['protonated charge']

print (lin_reg.score(X, Y))

#text=data["compn"]
text=[n+" ("+"%+d" %c+" -> "+"%+d" %(c-1)+")" for n,c in zip(data['correct name'],data['protonated charge'])]

scatter=go.Scatter(
                               y=pka_prediction,x=Y,mode='markers',text=text,showlegend=False,
                                marker=dict(color=data["protonated charge"], colorscale='Rainbow',cmin=-5, cmax=3,
                                line=dict(width=0.5),showscale=True, size=4,
                                colorbar=dict( y=0.8, x=0.15, orientation="h", title={"text": "charge of AH species","side":"top"},
                                                tickvals=list(range(-5,4)),
                                                #ticktext=["{:.2f}".format(i) for i in range(-5,3)],
                                                #tickness=18, 
                                              len=0.25
                                             )
                                           )
                        )
line=go.Scatter(y=[-8, 16],x=[-8,16],mode="lines",line=dict(color='black', width=3,dash='dash'),showlegend=False)
shadow05=go.Scatter(y=[-8.5,-9.5,18.5,19.5],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.3)',
                          line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo='skip')
shadow1=go.Scatter(y=[-8.0,-10.0,18.0,20.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.2)',
                          line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo='skip') 
shadow2=go.Scatter(y=[-7.0,-11.0,17.0,21.0],x=[-9,-9,19,19],fill="toself",fillcolor='rgba(0,80,80,0.1)',
                          line_color='rgba(255,255,255,0)',showlegend=False,hoverinfo='skip') 

fig1=go.Figure(data=[scatter,line,shadow05,shadow1,shadow2])
fig1.update_xaxes(visible=True,range=[-4,16],linewidth=2,linecolor="#444",showticklabels=True,title_text="exp. pKa",mirror=True)
fig1.update_yaxes(visible=True,range=[-4,16],linewidth=2,linecolor="#444",showticklabels=True,title_text="calc. pKa",mirror=True)
fig1.update_layout(height=800)

fig1.write_image("deltaGvspka.png", width=1000, height=800,scale=8)



residuals_histogram=go.Histogram(x=(Y-pka_prediction),opacity=0.75,xbins={"size":0.25},showlegend=False,
                                       marker_color="red",name="final model",legendgroup=1,legendrank=2,
                                    hoverinfo='skip'
                                    #xaxis='x2',yaxis='y2',

                                )


rug_residuals=go.Scatter(x=(Y-pka_prediction),y=[1.0]*len(Y-pka_prediction),mode='markers', text=text,showlegend=False,
                                   #xaxis='x2',yaxis='y2',
                                marker=dict(color=data["protonated charge"], colorscale='Rainbow',cmin=-5, cmax=3,
                                            symbol=142,size=20)          
                        )



fig2 = make_subplots( rows=2,cols=1,
                         #subplot_titles=["mue is:"+str(mean_unsigned_error),"residuals","errors"],
                         #subplot_titles=["histogram",],
                         row_heights=[0.8,0.2],
                         vertical_spacing=0.1,horizontal_spacing=0.0)
fig2.add_trace(residuals_histogram,row=1,col=1)
fig2.add_trace(rug_residuals,row=2,col=1)

fig2.update_layout(height=400,width=400,plot_bgcolor='rgba(255,255,255,1)',bargap=0.2)
#yaxis_range=[0,600],xaxis_range=[-4,4],
#paper_bgcolor='rgba(0,0,0,0)',
                

#xaxis2=dict(
#    domain=[0.7, 0.95],
#    anchor='y2'
#    ),
#yaxis2=dict(
#    domain=[0.05, 0.4],
#    anchor='x2'
#    )


fig2.update_xaxes(visible=True,row=1,col=1,range=[-4,4],griddash="dot",linewidth=2,
                  linecolor="#444",gridcolor="#D62728",showticklabels=True,mirror=True,tickvals=np.arange(-4,5,1))
fig2.update_yaxes(visible=True,row=1,col=1,griddash="dot",linewidth=2,linecolor="#444",gridcolor="#D62728",mirror=True)
fig2.update_xaxes(visible=False,row=2,col=1,range=[-4,4])
fig2.update_yaxes(visible=False,row=2,col=1,griddash="dot")



fig2.write_image("deltaGvspka-hist.png", width=400, height=400,scale=4)




abs_errors=abs(pka_prediction-Y)
mue=np.average(abs_errors)
print ("mue:"+str(np.average(abs_errors)))



print (lin_reg.get_params())
"""


def scattermatrix(data,upper_features,lower_features="",labels="",height=1200,width=1200,standarize_hist=False,outfile="",show=True):

    if lower_features=="": lower_features=upper_features

    if labels!="": labels=data[labels]
    else: labels=[]
    
    titles=[""]
    for i in range(len(upper_features)): titles.append(transl_symbols(upper_features[i]))
    titles.append("")

    for i in range(0,len(upper_features)):
        titles.append(transl_symbols(lower_features[i]))
        for  j in range(0,len(lower_features)):titles.append("")
        titles.append(transl_symbols(upper_features[i]))
    titles.append("")
    for i in range(len(upper_features)): titles.append(transl_symbols(lower_features[i]))
    titles.append("")
    #print (titles)
    row_heights=[0.0]+[0.9/len(upper_features) for _ in range(len(upper_features))]+[0.15]
    column_widths=[0.0]+[0.9/len(upper_features) for _ in range(len(upper_features))]+[0.15]
    sctplot=make_subplots(rows=len(upper_features)+2, cols=len(upper_features)+2,horizontal_spacing=0.025, vertical_spacing=0.025, 
                          subplot_titles=titles,
                          row_heights=row_heights,column_widths=column_widths)

    marker=dict(color=data["protonated charge"], colorscale='Rainbow',cmin=-5, cmax=3,
                                        line=dict(width=0.5),size=2,line_width=0.5,
                                         )
    #for annotation in sctplot['layout']['annotations'][len(upper_features)+1::2]: print(annotation)

    for annotation in sctplot['layout']['annotations'][0:len(upper_features)]:    #[1:len(upper_features)+1] : 
            fontsize=min([28,1.4*width/((len(upper_features)+1)*(len(annotation["text"])+0.1))])
            font=plotly.graph_objects.layout.annotation.Font(size=fontsize,weight=1000)
            annotation['font']=font
            annotation['textangle']=-30.0
            annotation['xanchor']="center"
            annotation['yanchor']="bottom"
            #annotation["bgcolor"]="yellow"
            annotation['yshift']=-int(0.95*0.1*height/len(upper_features))
            #annotation['xshift']=-int(0.25*width/len(upper_features))
        
    for annotation in sctplot['layout']['annotations'][-len(upper_features):]: #[-(len(upper_features)+1):-1]:
            fontsize=min([28,1.4*width/((len(upper_features)+1)*(len(annotation["text"])+0.1))])
            font=plotly.graph_objects.layout.annotation.Font(size=fontsize,weight=1000)
            annotation['font']=font
            annotation['textangle']=-30.0
            annotation['xanchor']="center"
            annotation['yanchor']="top"
            #annotation["bgcolor"]="blue"
            annotation['yshift']=int(0.95*0.1*height/len(upper_features))
            #annotation['xshift']=-int(0.25*width/len(upper_features))
        
    for annotation in sctplot['layout']['annotations'][len(upper_features):-len(upper_features)-1:2]:#[len(upper_features)+2::len(upper_features)+2]:
            fontsize=min([28,1.4*width/((len(upper_features)+1)*(len(annotation["text"])+0.1))])
            font=plotly.graph_objects.layout.annotation.Font(size=fontsize,weight=1000)
            annotation['font']=font
            annotation['textangle']=-30
            annotation['xanchor']="right"
            annotation['yanchor']="middle"
            #annotation['bgcolor']="red"
            #annotation['xref']="x domain"
            #annotation['yref']="paper"
            annotation['yshift']=-int(0.95*0.4*height/len(upper_features))
            annotation['xshift']=int(0.10*width/len(upper_features))

    for annotation in sctplot['layout']['annotations'][len(upper_features)+1:-len(upper_features):2]:#[2*(len(upper_features)+2)-1::len(upper_features)+2]:
            fontsize=min([28,1.4*width/((len(upper_features)+1)*(len(annotation["text"])+0.1))])
            font=plotly.graph_objects.layout.annotation.Font(size=fontsize,weight=1000)
            annotation['font']=font
            annotation['textangle']=-30
            annotation['xanchor']="left"
            annotation['yanchor']="middle"
            #annotation['bgcolor']="green"
            #annotation['xref']="x domain"
            #annotation['yref']="paper"
            annotation['yshift']=-int(0.95*0.4*height/len(upper_features))
            annotation['xshift']=-int(0.4*width/len(upper_features))


    if standarize_hist: 
        
        std_data={}
        for f in upper_features:
            print ("standarizing... "+f,end="\r")
            #t=np.asarray(data[f]).reshape(-1,1)
            std_data[f]= ((data[f]-np.mean(data[f]))/np.std(data[f]))  
        if upper_features!=lower_features:
            for f in lower_features:
                print ("standarizing... "+f,end="\r")
                #t=np.asarray(data[f]).reshape(-1,1)
                #std_data[f]=sklearn.preprocessing.StandardScaler().fit_transform(t) 
                std_data[f]= ((data[f]-np.mean(data[f]))/np.std(data[f]))
    else: 
        std_data=data

    hist_data={}
    for f,f2 in zip(upper_features,lower_features):
        print ("histograming..."+f,end="\r")
        hist=np.histogram(std_data[f],bins=500)
        #print (hist)
        hist_data[f]=[hist[0]/np.max(hist[0]),(hist[1][:-1]-np.mean(hist[1]))/(np.max(hist[1])-np.min(hist[1]))] #last bin value is removed  and values are normalized
        #print (hist_data)
      
        rep_bins=np.repeat(hist_data[f][1],3)
        rep_counts=np.repeat(hist_data[f][0],3)
        rep_counts[0::3]=0
        rep_counts[2::3]=0
        rep_counts=rep_counts/np.max(rep_counts)
        hist_data[f]=[rep_counts,rep_bins]
        #print (hist_data)
        if f!=f2:  
            #rotate data:
            #hist_data[f][0]=0.5*hist_data[f][0]
            #hist_data[f][0],hist_data[f][1]=hist_data[f][0]*np.sqrt(2)/2-hist_data[f][1]*np.sqrt(2)/2,hist_data[f][0]*np.sqrt(2)/2+hist_data[f][1]*np.sqrt(2)/2
            #hist_data[f][0],hist_data[f][1]=0.5+hist_data[f][1],0.5-hist_data[f][0]
            hist_data[f][0],hist_data[f][1]=1-hist_data[f][0],hist_data[f][1]
                   
            print ("histograming..."+f2,end="\r")
            hist=np.histogram(std_data[f2],bins=500)
            hist_data[f2]=[hist[0]/np.max(hist[0]),(hist[1][:-1]-np.mean(hist[1]))/(np.max(hist[1])-np.min(hist[1]))] #last bin value is removed  and values are normalized
            rep_bins=np.repeat(hist_data[f2][1],3)  #change the order
            rep_counts=np.repeat(hist_data[f2][0],3)
            rep_counts[0::3]=0
            rep_counts[2::3]=0
            rep_counts=rep_counts/np.max(rep_counts)
            hist_data[f2]=[rep_counts,rep_bins] 
            #hist_data[f2][0]=2*hist_data[f2][0]
            
            #rotate data:
            #hist_data[f2][0]=rep_counts*np.sqrt(2)/2-rep_bins*np.sqrt(2)/2
            #hist_data[f2][1]=rep_counts*np.sqrt(2)/2+rep_bins*np.sqrt(2)/2
            hist_data[f2][0]=rep_counts
            hist_data[f2][1]=rep_bins

        
        

    
    for i in range(len(upper_features)):
           
        if upper_features[i]==lower_features[i]:
            print ("generating histogram for: "+  upper_features[i] ) 

            hist=go.Scatter(y=hist_data[upper_features[i]][0],x=hist_data[upper_features[i]][1],mode="lines",
                            opacity=1,showlegend=False,line_color="dodgerblue",hoverinfo='skip',
                           )
            line=go.Scatter(y=[0,0],x=[-0.5,0.5],mode="lines",opacity=1,line_color="black",showlegend=False,hoverinfo='skip')     
            sctplot.add_trace(hist,row=2+i, col=2+i)
            sctplot.add_trace(line,row=2+i, col=2+i)
        else:
            print ("generating histogram for: "+  upper_features[i] ) 

            hist1=go.Scatter(y=hist_data[upper_features[i]][0],x=hist_data[upper_features[i]][1],mode="lines",
                            opacity=1,showlegend=False,line_color="dodgerblue",hoverinfo='skip',
                           )
            
            print ("generating histogram for: "+  lower_features[i] ) 

            hist2=go.Scatter(y=hist_data[lower_features[i]][0],x=hist_data[lower_features[i]][1],mode="lines",
                            opacity=0.7,showlegend=False,line_color="red",hoverinfo='skip',
                           )
            print ("done: hist "+lower_features[i] )
            #draw diagonal line
            upper_max_x,upper_min_x=np.max(hist_data[upper_features[i]][1]),np.min(hist_data[upper_features[i]][1])
            lower_max_x,lower_min_x=np.max(hist_data[lower_features[i]][1]),np.min(hist_data[lower_features[i]][1])
            X=np.max([upper_max_x,lower_max_x,-upper_min_x,-lower_min_x])
            X1=np.max([upper_max_x,lower_max_x])
            X2=np.min([upper_min_x,lower_min_x])
            upper_max_y,upper_min_y=np.max(hist_data[upper_features[i]][0]),np.min(hist_data[upper_features[i]][0])
            lower_max_y,lower_min_y=np.max(hist_data[lower_features[i]][0]),np.min(hist_data[lower_features[i]][0])
            Y=np.max([upper_max_y,lower_max_y,-upper_min_y,-lower_min_y])
            Y1=np.max([upper_max_y,lower_max_y])
            Y2=np.min([upper_min_y,lower_min_y])
            Z=np.max([X,Y])
            
            line=go.Scatter(y=[-Z,Z],x=[Z,-Z],mode="lines",opacity=1,line_color="black",showlegend=False,hoverinfo='skip')
            sctplot.add_trace(hist1,row=2+i, col=2+i)
            sctplot.add_trace(hist2,row=2+i, col=2+i)
            #sctplot.add_trace(line,row=2+i, col=2+i)
            
    
        for j in range(i+1,len(upper_features)):
            print ("generating scatter plot for:" +upper_features[i]+ " and "+ upper_features[j])
            sct=go.Scatter(y=data[upper_features[i]], x=data[upper_features[j]],mode="markers",showlegend=False,
                            marker=marker,text=labels
                                         ) 

            print ("generating scatter plot for:" +lower_features[i]+ " and "+ lower_features[j])
            sct2=go.Scatter(y=data[lower_features[j]], x=data[lower_features[i]],mode="markers",showlegend=False,
                            marker=marker,text=labels
                                         ) 

            
            #sct.update_axis(visible=False)
            sctplot.add_trace(sct,row=2+i, col=2+j)
            sctplot.add_trace(sct2,row=2+j, col=2+i)

        
            sctplot.update_xaxes(visible=True,showticklabels=False, showgrid=True,showline=True,row=2+i, col=2+j,zeroline=True,zerolinecolor="lightblue",zerolinewidth=2,
                                    gridcolor="aliceblue",gridwidth=10,nticks=1000) 
            sctplot.update_yaxes(visible=True,showticklabels=False, showgrid=False,showline=True,row=2+i, col=2+j,zeroline=True,zerolinecolor="lightblue",zerolinewidth=2,
                                    )
            if upper_features[i]==lower_features[i] and upper_features==lower_features:
                sctplot.update_xaxes(visible=True,showticklabels=False, showgrid=True,showline=True,row=2+j, col=2+i,zeroline=True,zerolinecolor="lightblue",zerolinewidth=2,
                                     gridcolor="aliceblue",gridwidth=10,nticks=1000) 
                sctplot.update_yaxes(visible=True,showticklabels=False, showgrid=False,showline=True,row=2+j, col=2+i,zeroline=True,zerolinecolor="lightblue",zerolinewidth=2,
                                    )
            else:
                sctplot.update_xaxes(visible=True,showticklabels=False, showgrid=True,showline=True,row=2+j, col=2+i,zeroline=True,zerolinecolor="gold",zerolinewidth=2,
                                gridcolor="ivory",gridwidth=10,nticks=1000)       
                sctplot.update_yaxes(visible=True,showticklabels=False, showgrid=False,showline=True,row=2+j, col=2+i,zeroline=True,zerolinecolor="gold",zerolinewidth=2,
                                    )

        sctplot.update_xaxes(visible=False,showticklabels=False, showgrid=True,showline=False,row=2+i, col=2+i,zeroline=False,gridcolor="gainsboro",gridwidth=10,nticks=1000,
                            )       
        sctplot.update_yaxes(visible=False,showticklabels=False, showgrid=False,showline=False,row=2+i, col=2+i,zeroline=False,gridcolor="gainsboro",
                            )
                    
        
    
    sctplot.update_layout(height=height,width=width,title_text="",plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0.8*width/len(upper_features),
                                      r=0.1*width/len(upper_features),
                                      b=0.1*width/len(upper_features),
                                      t=0.6*width/len(upper_features),
                                      pad=4),
                         
                         )
    if outfile!="":
        print ("writing files")
        sctplot.write_image("./"+outfile+".png", width=1200, height=1200,scale=3)
        sctplot.write_html("./"+outfile+".html")

    if show:
        print ("showing") 
        sctplot.show()
"""    
descriptors=["pKa","deltaG","deltaZPE","deltaE","SMD-solv","expl1wat","NBO-HB"]
scattermatrix(data=data,upper_features=descriptors,lower_features="",labels="compn",outfile="energies_scatterplot",show=False)

RDG=["pKa","RDG%HB","RDG%VdW","RDG%st","prom-RDG%HB","prom-RDG%VdW","prom-RDG%st"]
moments=["pKa","HLgap","*mu","tr-*theta","tr-e*theta","tr-*alpha","Vol"]
scattermatrix(data=data,upper_features=RDG,lower_features=moments,labels="compn",outfile="RDG_moments_scatterplot",show=False)


ESP=["pKa","max-ESP","min-ESP","avg-ESP","var-ESP","*PI-ESP","MPI"]
ESP_sign=["pKa","avg-ESP+","var-ESP+","avg-ESP-","var-ESP-","Surf+","Surf-"]
scattermatrix(data=data,upper_features=ESP,lower_features=ESP_sign,labels="compn",outfile="ESP_scatterplot",show=False)

ALIE=["pKa","max-ALIE","min-ALIE","avg-ALIE","var-ALIE"]
LEA=["pKa","max-LEA","min-LEA","avg-LEA","var-LEA"]
scattermatrix(data=data,upper_features=ALIE,lower_features=LEA,labels="compn",outfile="ALIE_LEA_scatterplot",show=False)


ESP_ALIE=["pKa","max-ESP","min-ESP","avg-ESP","var-ESP","*PI-ESP","MPI","max-ALIE","min-ALIE","avg-ALIE","var-ALIE"]
ESP_sign_LEA=["pKa","avg-ESP+","var-ESP+","avg-ESP-","var-ESP-","Surf+","Surf-","max-LEA","min-LEA","avg-LEA","var-LEA"]
scattermatrix(data=data,upper_features=ESP_ALIE,lower_features=ESP_sign_LEA,labels="compn",outfile="ESP_ALIE_LEA_scatterplot",show=False)

EF1=["pKa","0.95q|EF|","0.9q|EF|","0.75q|EF|","avg|EF|","0.95qEF*norm","0.9qEF*norm","0.75qEF*norm","avgEF*norm"]
EF2=["pKa","0.95qEF*tang","0.9qEF*tang","0.75qEF*tang","avgEF*tang","0.95qEF*angle","0.9qEF*angle","0.75qEF*angle","avgEF*angle"]
scattermatrix(data=data,upper_features=EF1,lower_features=EF2,labels="compn",outfile="EF_scatterplot",show=False)

EF3=["pKa","0.95q|EF|","avg|EF|","0.95qEF*norm","avgEF*norm","0.95qEF*tang","avgEF*tang","0.95qEF*angle","avgEF*angle"]
EF4=["pKa","0.9q|EF|","0.75q|EF|","0.9qEF*norm","0.75qEF*norm","0.9qEF*tang","0.75qEF*tang","0.9qEF*angle","0.75qEF*angle"]
scattermatrix(data=data,upper_features=EF3,lower_features=EF4,labels="compn",outfile="EF_scatterplot2",show=False)

charges_alpha_1=["pKa","PEOE_alpha",
               "Mulliken_alpha","Lowdin_alpha",
               "Hirshfeld_alpha","Voronoy_alpha","Becke_alpha","ESP-nucl_alpha"
                ]
charges_alpha_2=["pKa",
               "ADCH_alpha","CM5_alpha","12CM5_alpha",
               "CHELPG_alpha","MK_alpha","RESP_alpha",
               "NBO-chg_alpha"]
scattermatrix(data=data,upper_features=charges_alpha_1,lower_features=charges_alpha_2,labels="compn",outfile="chg_alpha_scatterplot",show=False)

charges_prot_alpha_1=["pKa","protonated PEOE_alpha",
               "protonated Mulliken_alpha","protonated Lowdin_alpha",
               "protonated Hirshfeld_alpha","protonated Voronoy_alpha","protonated Becke_alpha","protonated ESP-nucl_alpha"
                ]
charges_prot_alpha_2=["pKa",
               "protonated ADCH_alpha","protonated CM5_alpha","protonated 12CM5_alpha",
               "protonated CHELPG_alpha","protonated MK_alpha","protonated RESP_alpha",
               "protonated NBO-chg_alpha"]
scattermatrix(data=data,upper_features=charges_prot_alpha_1,lower_features=charges_prot_alpha_2,labels="compn",outfile="protoanted_chg_alpha_scatterplot",show=False)


charges_deprot_alpha_1=["pKa","deprotonated PEOE_alpha",
               "deprotonated Mulliken_alpha","deprotonated Lowdin_alpha",
               "deprotonated Hirshfeld_alpha","deprotonated Voronoy_alpha","deprotonated Becke_alpha","deprotonated ESP-nucl_alpha"
                ]
charges_deprot_alpha_2=["pKa",
               "deprotonated ADCH_alpha","deprotonated CM5_alpha","deprotonated 12CM5_alpha",
               "deprotonated CHELPG_alpha","deprotonated MK_alpha","deprotonated RESP_alpha",
               "deprotonated NBO-chg_alpha"]
scattermatrix(data=data,upper_features=charges_deprot_alpha_1,lower_features=charges_deprot_alpha_2,labels="compn",outfile="deprotoanted_chg_alpha_scatterplot",show=False)

charges_beta_1=["pKa","PEOE_beta",
               "Mulliken_beta","Lowdin_beta",
               "Hirshfeld_beta","Voronoy_beta","Becke_beta","ESP-nucl_beta"

               ]
charges_beta_2=["pKa",
               "ADCH_beta","CM5_beta","12CM5_beta",
               "CHELPG_beta","MK_beta","RESP_beta",
               "NBO-chg_beta",
               ]
scattermatrix(data=data,upper_features=charges_beta_1,lower_features=charges_beta_2,labels="compn",outfile="chg_beta_scatterplot2",show=False)


charges_protonated_beta_1=["pKa","protonated PEOE_beta",
               "protonated Mulliken_beta","protonated Lowdin_beta",
               "protonated Hirshfeld_beta","protonated Voronoy_beta","protonated Becke_beta","protonated ESP-nucl_beta"

               ]
charges_protonated_beta_2=["pKa",
               "protonated ADCH_beta","protonated CM5_beta","protonated 12CM5_beta",
               "protonated CHELPG_beta","protonated MK_beta","protonated RESP_beta",
               "protonated NBO-chg_beta",
               ]
scattermatrix(data=data,upper_features=charges_protonated_beta_1,lower_features=charges_protonated_beta_2,labels="compn",outfile="protonated_chg_beta_scatterplot2",show=False)




charges_deprotonated_beta_1=["pKa","deprotonated PEOE_beta",
               "deprotonated Mulliken_beta","deprotonated Lowdin_beta",
               "deprotonated Hirshfeld_beta","deprotonated Voronoy_beta","deprotonated Becke_beta","deprotonated ESP-nucl_beta"

               ]
charges_deprotonated_beta_2=["pKa",
               "deprotonated ADCH_beta","deprotonated CM5_beta","deprotonated 12CM5_beta",
               "deprotonated CHELPG_beta","deprotonated MK_beta","deprotonated RESP_beta",
               "deprotonated NBO-chg_beta",
               ]
scattermatrix(data=data,upper_features=charges_deprotonated_beta_1,lower_features=charges_deprotonated_beta_2,labels="compn",outfile="deprotonated_chg_beta_scatterplot2",show=False)

#FALTA
ESP_alpha=["pKa",
            "(a)avg-ESP_alpha",
            "(a)avg-ESP+_alpha",
            "(a)avg-ESP-_alpha",
            "(a)var-ESP_alpha",
            "(a)var-ESP+_alpha",
            "(a)var-ESP-_alpha",
            "(a)min-ESP_alpha",
            "(a)max-ESP_alpha"]
ESP_ALIELEA_alpha=[ "pKa",
               "(a)avg-ALIE_alpha","(a)var-ALIE_alpha","(a)max-ALIE_alpha","(a)min-ALIE_alpha",
               "(a)avg-LEA_alpha","(a)var-LEA_alpha","(a)max-LEA_alpha","(a)min-LEA_alpha"]


mu_alpha=["pKa",           
               "(a)*mu_alpha","(a)*mu-ctb_alpha","(a)tr-e*theta_alpha","NMR*delta_alpha",
         ]

surf_alpha=["pKa",
            "(a)Surf_alpha",
            "(a)Surf-_alpha",
            "(a)Surf+_alpha",
            "(a)*PI-ESP_alpha"]

scattermatrix(data=data,upper_features=ESP_alpha,lower_features=ESP_ALIELEA_alpha,labels="compn",outfile="ESP_alpha_scatterplot",show=False)

scattermatrix(data=data,upper_features=mu_alpha,lower_features=surf_alpha,labels="compn",outfile="mu_alpha_scatterplot",show=False)



protonated_ESP_alpha=["pKa",
            "protonated (a)avg-ESP_alpha",
            "protonated (a)avg-ESP+_alpha",
            "protonated (a)avg-ESP-_alpha",
            "protonated (a)var-ESP_alpha",
            "protonated (a)var-ESP+_alpha",
            "protonated (a)var-ESP-_alpha",
            "protonated (a)min-ESP_alpha",
            "protonated (a)max-ESP_alpha"]
protonated_ESP_ALIELEA_alpha=[ "pKa",
               "protonated (a)avg-ALIE_alpha","protonated (a)var-ALIE_alpha","protonated (a)max-ALIE_alpha","protonated (a)min-ALIE_alpha",
               "protonated (a)avg-LEA_alpha","protonated (a)var-LEA_alpha","protonated (a)max-LEA_alpha","protonated (a)min-LEA_alpha"]


protonated_mu_alpha=["pKa",           
               "protonated (a)*mu_alpha","protonated (a)*mu-ctb_alpha","protonated (a)tr-e*theta_alpha","protonated NMR*delta_alpha",
         ]

protonated_surf_alpha=["pKa",
            "protonated (a)Surf_alpha",
            "protonated (a)Surf-_alpha",
            "protonated (a)Surf+_alpha",
            "protonated (a)*PI-ESP_alpha"]

scattermatrix(data=data,upper_features=protonated_ESP_alpha,lower_features=protonated_ESP_ALIELEA_alpha,labels="compn",outfile="protonated_ESP_alpha_scatterplot",show=False)

scattermatrix(data=data,upper_features=protonated_mu_alpha,lower_features=protonated_surf_alpha,labels="compn",outfile="protonated_mu_alpha_scatterplot",show=False)



deprotonated_ESP_alpha=["pKa",
            "deprotonated (a)avg-ESP_alpha",
            "deprotonated (a)avg-ESP+_alpha",
            "deprotonated (a)avg-ESP-_alpha",
            "deprotonated (a)var-ESP_alpha",
            "deprotonated (a)var-ESP+_alpha",
            "deprotonated (a)var-ESP-_alpha",
            "deprotonated (a)min-ESP_alpha",
            "deprotonated (a)max-ESP_alpha"]
deprotonated_ESP_ALIELEA_alpha=[ "pKa",
               "deprotonated (a)avg-ALIE_alpha","deprotonated (a)var-ALIE_alpha","deprotonated (a)max-ALIE_alpha","deprotonated (a)min-ALIE_alpha",
               "deprotonated (a)avg-LEA_alpha","deprotonated (a)var-LEA_alpha","deprotonated (a)max-LEA_alpha","deprotonated (a)min-LEA_alpha"]


deprotonated_mu_alpha=["pKa",           
               "deprotonated (a)*mu_alpha","deprotonated (a)*mu-ctb_alpha","deprotonated (a)tr-e*theta_alpha","deprotonated NMR*delta_alpha",
         ]

deprotonated_surf_alpha=["pKa",
            "deprotonated (a)Surf_alpha",
            "deprotonated (a)Surf-_alpha",
            "deprotonated (a)Surf+_alpha",
            "deprotonated (a)*PI-ESP_alpha"]

scattermatrix(data=data,upper_features=deprotonated_ESP_alpha,lower_features=deprotonated_ESP_ALIELEA_alpha,labels="compn",outfile="deprotonated_ESP_alpha_scatterplot",show=False)

scattermatrix(data=data,upper_features=deprotonated_mu_alpha,lower_features=deprotonated_surf_alpha,labels="compn",outfile="deprotonated_mu_alpha_scatterplot",show=False)



ESP_beta=["pKa",
            "(a)avg-ESP_beta",
            "(a)avg-ESP+_beta",
            "(a)avg-ESP-_beta",
            "(a)var-ESP_beta",
            "(a)var-ESP+_beta",
            "(a)var-ESP-_beta",
            "(a)min-ESP_beta",
            "(a)max-ESP_beta"]
ESP_ALIELEA_beta=[ "pKa",
               "(a)avg-ALIE_beta","(a)var-ALIE_beta","(a)max-ALIE_beta","(a)min-ALIE_beta",
               "(a)avg-LEA_beta","(a)var-LEA_beta","(a)max-LEA_beta","(a)min-LEA_beta"]


mu_beta=["pKa",           
               "(a)*mu_beta","(a)*mu-ctb_beta","(a)tr-e*theta_beta","NMR*delta_beta",
         ]

surf_beta=["pKa",
            "(a)Surf_beta",
            "(a)Surf-_beta",
            "(a)Surf+_beta",
            "(a)*PI-ESP_beta"]

scattermatrix(data=data,upper_features=ESP_alpha,lower_features=ESP_ALIELEA_beta,labels="compn",outfile="ESP_beta_scatterplot",show=False)

scattermatrix(data=data,upper_features=mu_alpha,lower_features=surf_beta,labels="compn",outfile="mu_beta_scatterplot",show=False)


protonated_ESP_beta=["pKa",
            "protonated (a)avg-ESP_beta",
            "protonated (a)avg-ESP+_beta",
            "protonated (a)avg-ESP-_beta",
            "protonated (a)var-ESP_beta",
            "protonated (a)var-ESP+_beta",
            "protonated (a)var-ESP-_beta",
            "protonated (a)min-ESP_beta",
            "protonated (a)max-ESP_beta"]
protonated_ESP_ALIELEA_beta=[ "pKa",
               "protonated (a)avg-ALIE_beta","protonated (a)var-ALIE_beta","protonated (a)max-ALIE_beta","protonated (a)min-ALIE_beta",
               "protonated (a)avg-LEA_beta","protonated (a)var-LEA_beta","protonated (a)max-LEA_beta","protonated (a)min-LEA_beta"]


protonated_mu_beta=["pKa",           
               "protonated (a)*mu_beta","protonated (a)*mu-ctb_beta","protonated (a)tr-e*theta_beta","protonated NMR*delta_beta",
         ]

protonated_surf_beta=["pKa",
            "protonated (a)Surf_beta",
            "protonated (a)Surf-_beta",
            "protonated (a)Surf+_beta",
            "protonated (a)*PI-ESP_beta"]

scattermatrix(data=data,upper_features=protonated_ESP_beta,lower_features=protonated_ESP_ALIELEA_beta,labels="compn",outfile="protonated_ESP_beta_scatterplot",show=False)

scattermatrix(data=data,upper_features=protonated_mu_beta,lower_features=protonated_surf_beta,labels="compn",outfile="protonated_mu_beta_scatterplot",show=False)



deprotonated_ESP_beta=["pKa",
            "deprotonated (a)avg-ESP_beta",
            "deprotonated (a)avg-ESP+_beta",
            "deprotonated (a)avg-ESP-_beta",
            "deprotonated (a)var-ESP_beta",
            "deprotonated (a)var-ESP+_beta",
            "deprotonated (a)var-ESP-_beta",
            "deprotonated (a)min-ESP_beta",
            "deprotonated (a)max-ESP_beta"]
deprotonated_ESP_ALIELEA_beta=[ "pKa",
               "deprotonated (a)avg-ALIE_beta","deprotonated (a)var-ALIE_beta","deprotonated (a)max-ALIE_beta","deprotonated (a)min-ALIE_beta",
               "deprotonated (a)avg-LEA_beta","deprotonated (a)var-LEA_beta","deprotonated (a)max-LEA_beta","deprotonated (a)min-LEA_beta"]


deprotonated_mu_beta=["pKa",           
               "deprotonated (a)*mu_beta","deprotonated (a)*mu-ctb_beta","deprotonated (a)tr-e*theta_beta","deprotonated NMR*delta_beta",
         ]

deprotonated_surf_beta=["pKa",
            "deprotonated (a)Surf_beta",
            "deprotonated (a)Surf-_beta",
            "deprotonated (a)Surf+_beta",
            "deprotonated (a)*PI-ESP_beta"]

scattermatrix(data=data,upper_features=deprotonated_ESP_beta,lower_features=deprotonated_ESP_ALIELEA_beta,labels="compn",outfile="deprotonated_ESP_beta_scatterplot",show=False)

scattermatrix(data=data,upper_features=deprotonated_mu_beta,lower_features=deprotonated_surf_beta,labels="compn",outfile="deprotonated_mu_beta_scatterplot",show=False)


deprotonated_ESP_beta=["pKa",
            "deprotonated (a)avg-ESP_beta",
            "deprotonated (a)avg-ESP+_beta",
            "deprotonated (a)avg-ESP-_beta",
            "deprotonated (a)var-ESP_beta",
            "deprotonated (a)var-ESP+_beta",
            "deprotonated (a)var-ESP-_beta",
            "deprotonated (a)min-ESP_beta",
            "deprotonated (a)max-ESP_beta"]
deprotonated_ESP_ALIELEA_beta=[ "pKa",
               "deprotonated (a)avg-ALIE_beta","deprotonated (a)var-ALIE_beta","deprotonated (a)max-ALIE_beta","deprotonated (a)min-ALIE_beta",
               "deprotonated (a)avg-LEA_beta","deprotonated (a)var-LEA_beta","deprotonated (a)max-LEA_beta","deprotonated (a)min-LEA_beta"]


deprotonated_mu_beta=["pKa",           
               "deprotonated (a)*mu_beta","deprotonated (a)*mu-ctb_beta","deprotonated (a)tr-e*theta_beta","deprotonated NMR*delta_beta",
         ]

deprotonated_surf_beta=["pKa",
            "deprotonated (a)Surf_beta",
            "deprotonated (a)Surf-_beta",
            "deprotonated (a)Surf+_beta",
            "deprotonated (a)*PI-ESP_beta"]

scattermatrix(data=data,upper_features=deprotonated_ESP_beta,lower_features=deprotonated_ESP_ALIELEA_beta,labels="compn",outfile="deprotonated_ESP_beta_scatterplot",show=False)

scattermatrix(data=data,upper_features=deprotonated_mu_beta,lower_features=deprotonated_surf_beta,labels="compn",outfile="deprotonated_mu_beta_scatterplot",show=False)

deprotonated_ESP_beta=["pKa",
            "deprotonated (a)avg-ESP_beta",
            "deprotonated (a)avg-ESP+_beta",
            "deprotonated (a)avg-ESP-_beta",
            "deprotonated (a)var-ESP_beta",
            "deprotonated (a)var-ESP+_beta",
            "deprotonated (a)var-ESP-_beta",
            "deprotonated (a)min-ESP_beta",
            "deprotonated (a)max-ESP_beta"]
deprotonated_ESP_ALIELEA_beta=[ "pKa",
               "deprotonated (a)avg-ALIE_beta","deprotonated (a)var-ALIE_beta","deprotonated (a)max-ALIE_beta","deprotonated (a)min-ALIE_beta",
               "deprotonated (a)avg-LEA_beta","deprotonated (a)var-LEA_beta","deprotonated (a)max-LEA_beta","deprotonated (a)min-LEA_beta"]


deprotonated_mu_beta=["pKa",           
               "deprotonated (a)*mu_beta","deprotonated (a)*mu-ctb_beta","deprotonated (a)tr-e*theta_beta","deprotonated NMR*delta_beta",
         ]

deprotonated_surf_beta=["pKa",
            "deprotonated (a)Surf_beta",
            "deprotonated (a)Surf-_beta",
            "deprotonated (a)Surf+_beta",
            "deprotonated (a)*PI-ESP_beta"]

scattermatrix(data=data,upper_features=deprotonated_ESP_beta,lower_features=deprotonated_ESP_ALIELEA_beta,labels="compn",outfile="deprotonated_ESP_beta_scatterplot",show=False)

scattermatrix(data=data,upper_features=deprotonated_mu_beta,lower_features=deprotonated_surf_beta,labels="compn",outfile="deprotonated_mu_beta_scatterplot",show=False)


charges_H_1=["pKa","PEOE-*H_alpha",
               "Mulliken-*H_alpha","Lowdin-*H_alpha",
               "Hirshfeld-*H_alpha","Voronoy-*H_alpha","Becke-*H_alpha"]
charges_H_2=["pKa",
               "ADCH-*H_alpha","CM5-*H_alpha",#"12CM5_*H_alpha",
               "CHELPG-*H_alpha","MK-*H_alpha","RESP-*H_alpha",
               "NBO-chg-*H_alpha"]

charges_H_rel_1=["pKa","PEOE*relative*H_alpha",
               "Mulliken*relative*H_alpha","Lowdin*relative*H_alpha",
               "Hirshfeld*relative*H_alpha","Voronoy*relative*H_alpha","Becke*relative*H_alpha",
                ]
charges_H_rel_2=["pKa",
               "ADCH*relative*H_alpha","CM5*relative*H_alpha",#"12CM5_*relative*H_alpha",
               "CHELPG*relative*H_alpha","MK*relative*H_alpha","RESP*relative*H_alpha",
               "NBO-chg*relative*H_alpha"
              ]

scattermatrix(data=data,upper_features=charges_H_1,lower_features=charges_H_2,labels="compn",outfile="charges_H_scatterplot",show=False)

scattermatrix(data=data,upper_features=charges_H_rel_1,lower_features=charges_H_rel_2,labels="compn",outfile="charges_H_rel_scatterplot",show=False)
"""

XHbonds_rel_1=["pKa",
                "IBSI*relative*H", "Mayer-BO*relative*H", 
                "WBO*relative*H", "Mulliken-BO*relative*H", 
                "LBO*relative*H", "NLMO-BO*relative*H",
                "FUERZA-FC*relative*H","BD*relative*H"
                ]

#XHbonds_rel_2=["pKa",
#                "FUERZA-FC*relative*H","BD*relative*H","*mu*BP*relative*H","*ind*mu*BP*relative*H",
#                "diag-e*theta*BP*relative*H","diag-*theta*BP*relative*H"
#                ]
XHbonds_rel_2=XHbonds_rel_1


XHbonds_1=["pKa",
                "IBSI-*H", "Mayer-BO-*H", 
                "WBO-*H", "Mulliken-BO-*H", 
                "LBO-*H", "NLMO-BO-*H",
                ]

XHbonds_2=["pKa",
                "FUERZA-FC-*H","BD-*H","*mu*BP-*H","*ind*mu*BP-*H",
                "diag-e*theta*BP-*H","diag-*theta*BP-*H"
                ]

scattermatrix(data=data,upper_features=XHbonds_rel_1,lower_features=XHbonds_rel_2,labels="compn",outfile="BONDSH_rel_scatterplot",show=False)

scattermatrix(data=data,upper_features=XHbonds_1,lower_features=XHbonds_2,labels="compn",outfile="BONDSH_scatterplot",show=False)


from plotly.subplots import make_subplots

row_heights=[0.2,0.8]
column_widths=[0.8,0.2]
fig=make_subplots(rows=2, cols=2,horizontal_spacing=0.025, vertical_spacing=0.025, 
                          #subplot_titles=titles,
                          row_heights=row_heights,column_widths=column_widths)

text=[n+" ("+"%+d" %c+" -> "+"%+d" %(c-1)+")" for n,c in zip(data['correct name'],data['protonated charge'])]

sct=go.Scatter(
                                x=data["pKa"],y=data["MW"],mode='markers',text=text,showlegend=False,
                                marker=dict(color=data["protonated charge"], colorscale='Rainbow',cmin=-5, cmax=3,
                                line=dict(width=1),showscale=True, size=4,
                                colorbar=dict( y=0.8, x=0.9, orientation="h", title={"text": "charge of AH","side":"top"},title_font = dict(size=32),
                                                tickvals=list([-4,-2,0,1,2,3,4]),tickfont = dict(size=32),
                                                #ticktext=["{:.2f}".format(i) for i in range(-5,3)],
                                                #tickness=18, 
                                              len=0.25
                                             )
                                           )
                        )
pka_histogram=go.Histogram(x=data["pKa"],opacity=0.75,xbins={"size":0.25},showlegend=False,
                                       marker_color="dodgerblue",name="final model",legendgroup=1,legendrank=2,
                                    hoverinfo='skip'
                                    #xaxis='x2',yaxis='y2',
                                )
MW_histogram=go.Histogram(y=data["pKa"],opacity=0.75,xbins={"size":0.25},showlegend=False,
                                       marker_color="dodgerblue",name="final model",legendgroup=1,legendrank=2,
                                    hoverinfo='skip'
                                    #xaxis='x2',yaxis='y2',
                                )
fig.add_trace(sct,row=2,col=1)
fig.add_trace(pka_histogram,row=1,col=1)
fig.add_trace(MW_histogram,row=2,col=2)

fig.update_xaxes(visible=True,row=1,col=1,showticklabels=False, showgrid=True,gridcolor="white",gridwidth=10,nticks=1000)
fig.update_yaxes(visible=False,row=1,col=1)
fig.update_xaxes(visible=True,row=2,col=2,showticklabels=False, showgrid=True,gridcolor="white",gridwidth=10,nticks=1000)
fig.update_yaxes(visible=False,row=2,col=2)
fig.update_yaxes(visible=True,row=2,col=1,linewidth=2,linecolor="#444",showticklabels=True,title_text="Molecular Weight",
                 tickfont = dict(size=24),mirror=True, title_font = dict(size=40))
fig.update_xaxes(visible=True,row=2,col=1,linewidth=2,linecolor="#444",showticklabels=True,title_text="exp. pKa",
                 tickfont = dict(size=24),mirror=True, title_font = dict(size=40))


fig.update_layout(height=1200,width=1200,
                  #plot_bgcolor='rgba(255,255,255,1)',bargap=0.2
                 )

fig.write_image("MWvspka.png", width=1200, height=1200,scale=2)
fig.write_html("MWvspka.html")


import plotly.express as px
print(px.colors.sequential.Blues)
descriptors=["deltaG","deltaZPE","deltaE","SMD-solv","expl1wat","NBO-HB","RDG%HB","RDG%VdW","RDG%st","prom-RDG%HB","prom-RDG%VdW","prom-RDG%st"]
descriptors+=["max-ESP","min-ESP","avg-ESP","var-ESP","*PI-ESP","MPI","avg-ESP+","var-ESP+","avg-ESP-","var-ESP-","Surf","Surf+","Surf-"]
descriptors+=["max-ALIE","min-ALIE","avg-ALIE","var-ALIE","max-LEA","min-LEA","avg-LEA","var-LEA"]
descriptors+=["0.95q|EF|","0.9q|EF|","0.75q|EF|","avg|EF|","0.95qEF*norm","0.9qEF*norm","0.75qEF*norm","avgEF*norm",
              "0.95qEF*tang","0.9qEF*tang","0.75qEF*tang","avgEF*tang","0.95qEF*angle","0.9qEF*angle","0.75qEF*angle","avgEF*angle"]

translated_symbols=[transl_symbols(s) for s in descriptors]

d=data[descriptors].corr()

rename_dict={}
for dd,tt in zip(descriptors,translated_symbols): rename_dict[dd]=tt

d=d.rename(columns=rename_dict, index=rename_dict)


def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale 

bvals=  [0,0.2, 0.3,0.4, 0.5,0.6, 0.7, 0.8, 0.90,0.99,0.999]
colors=  ['rgb(228,240,252)', 'rgb(212,225,240)', 'rgb(198,219,239)', 'rgb(158,202,225)', 'rgb(107,174,214)', 'rgb(66,146,198)', 'rgb(33,113,181)', 'rgb(8,81,156)', 'rgb(8,48,107)','rgb(0,24,55)']
dcolorsc= discrete_colorscale(bvals, colors)
mask = np.triu(np.ones_like(d, dtype=bool))
d = d.mask(mask)
heat = go.Heatmap(
    z = np.square(d),
    x = d.columns.values,
    y = d.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    #tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    tickvals=[0.0,0.2,0.4,0.6, 0.8, 0.9,1],
                    orientation="h", title={"text": "r\u00b2","side":"top"},title_font=dict(size=16,weight=1000),
                    y=-0.2, x=0.0,len=0.25
                        #ticktext=ticktext)
                        )
)



title = 'Asset Correlation Matrix'

layout = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

fig=go.Figure(data=[heat], layout=layout)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=75,
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_eq.png", width=1200, height=1200,scale=4)
fig.write_html("corr_eq.html")


charges_alpha=["PEOE_alpha",
               "Mulliken_alpha","Lowdin_alpha",
               "Hirshfeld_alpha","Voronoy_alpha","Becke_alpha",
               "ADCH_alpha","CM5_alpha",#"12CM5_alpha",
               "CHELPG_alpha","MK_alpha","RESP_alpha",
               "NBO-chg_alpha"]
charges_beta=["PEOE_beta",
               "Mulliken_beta","Lowdin_beta",
               "Hirshfeld_beta","Voronoy_beta","Becke_beta",
               "ADCH_beta","CM5_beta",#"12CM5_beta",
               "CHELPG_beta","MK_beta","RESP_beta",
               "NBO-chg_beta"
               ]
ESP_alpha=[
               "(a)Surf_alpha","(a)Surf+_alpha","(a)Surf-_alpha",
               "(a)max-ESP_alpha","(a)min-ESP_alpha",
               "(a)avg-ESP_alpha","(a)avg-ESP+_alpha","(a)avg-ESP-_alpha",
               "(a)var-ESP_alpha","(a)var-ESP+_alpha","(a)var-ESP-_alpha",
               "(a)*PI-ESP_alpha"
               ]
ESP_beta=[
               "(a)Surf_beta","(a)Surf+_beta","(a)Surf-_beta",
               "(a)max-ESP_beta","(a)min-ESP_beta",
               "(a)avg-ESP_beta","(a)avg-ESP+_beta","(a)avg-ESP-_beta",
               "(a)var-ESP_beta","(a)var-ESP+_beta","(a)var-ESP-_beta",
               "(a)*PI-ESP_beta"
               ]
moments_alpha=[
               "ESP-nucl_alpha","NMR*delta_alpha",
               "(a)*mu_alpha","(a)*mu-ctb_alpha","(a)tr-e*theta_alpha",
               "(a)avg-ALIE_alpha","(a)var-ALIE_alpha","(a)max-ALIE_alpha","(a)min-ALIE_alpha",
               "(a)avg-LEA_alpha","(a)var-LEA_alpha","(a)max-LEA_alpha","(a)min-LEA_alpha"
                ]
moments_beta=[
               "ESP-nucl_beta","NMR*delta_alpha",
               "(a)*mu_beta","(a)*mu-ctb_beta","(a)tr-e*theta_beta",
               "(a)avg-ALIE_beta","(a)var-ALIE_beta","(a)max-ALIE_beta","(a)min-ALIE_beta",
               "(a)avg-LEA_beta","(a)var-LEA_beta","(a)max-LEA_beta","(a)min-LEA_beta"               
                ]
descriptors_alpha=charges_alpha+ESP_alpha+moments_alpha
descriptors_beta=charges_beta+ESP_beta+moments_beta
translated_symbols_alpha=[transl_symbols(s) for s in descriptors_alpha]
translated_symbols_beta=[transl_symbols(s) for s in descriptors_beta]



d_alpha=data[descriptors_alpha].corr()
rename_dict_alpha={}
for dd,tt in zip(descriptors_alpha,translated_symbols_alpha): rename_dict_alpha[dd]=tt
d_alpha=d_alpha.rename(columns=rename_dict_alpha, index=rename_dict_alpha)

d_beta=data[descriptors_beta].corr()
rename_dict_beta={}
for dd,tt in zip(descriptors_beta,translated_symbols_beta): rename_dict_beta[dd]=tt
d_beta=d_beta.rename(columns=rename_dict_beta, index=rename_dict_beta)


mask_u = np.triu(np.ones_like(d_alpha, dtype=bool))
mask_d = np.tril(np.ones_like(d_beta, dtype=bool))
d_alpha = d_alpha.mask(mask_d)
d_beta = d_beta.mask(mask_u)

heat_alpha = go.Heatmap(
    z = np.square(d_alpha),
    x = d_alpha.columns.values,
    y = d_alpha.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    orientation="h", title={"text": "r\u00b2","side":"top"},
                    y=0.5, x=0.2,len=0.25
                        #ticktext=ticktext)
                        )
)

heat_beta = go.Heatmap(
    z = np.square(d_beta),
    x = d_beta.columns.values,
    y = d_beta.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    orientation="h", title={"text": "r\u00b2","side":"top"},
                    y=0.5, x=0.7,len=0.25
                        #ticktext=ticktext)
                        )
)


title = 'Asset Correlation Matrix'

layout_a = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

layout_b = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

fig=go.Figure(data=[heat_beta], layout=layout_a)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=75,
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_beta.png", width=1200, height=1200,scale=4)
fig.write_html("corr_beta.html")


fig=go.Figure(data=[heat_alpha], layout=layout_b)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=75,side="top",
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",side="right",
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_alpha.png", width=1200, height=1200,scale=4)
fig.write_html("corr_alpha.html")



protonated_charges_alpha=["protonated PEOE_alpha",
               "protonated Mulliken_alpha","protonated Lowdin_alpha",
               "protonated Hirshfeld_alpha","protonated Voronoy_alpha","protonated Becke_alpha",
               "protonated ADCH_alpha","protonated CM5_alpha",#"12CM5_alpha",
               "protonated CHELPG_alpha","protonated MK_alpha","protonated RESP_alpha",
               "protonated NBO-chg_alpha"]
protonated_charges_beta=["protonated PEOE_beta",
               "protonated Mulliken_beta","protonated Lowdin_beta",
               "protonated Hirshfeld_beta","protonated Voronoy_beta","protonated Becke_beta",
               "protonated ADCH_beta","protonated CM5_beta",#"12CM5_beta",
               "protonated CHELPG_beta","protonated MK_beta","protonated RESP_beta",
               "protonated NBO-chg_beta"
               ]
protonated_ESP_alpha=[
               "protonated (a)Surf_alpha","protonated (a)Surf+_alpha","protonated (a)Surf-_alpha",
               "protonated (a)max-ESP_alpha","protonated (a)min-ESP_alpha",
               "protonated (a)avg-ESP_alpha","protonated (a)avg-ESP+_alpha","protonated (a)avg-ESP-_alpha",
               "protonated (a)var-ESP_alpha","protonated (a)var-ESP+_alpha","protonated (a)var-ESP-_alpha",
               "protonated (a)*PI-ESP_alpha"
               ]
protonated_ESP_beta=[
               "protonated (a)Surf_beta","protonated (a)Surf+_beta","protonated (a)Surf-_beta",
               "protonated (a)max-ESP_beta","protonated (a)min-ESP_beta",
               "protonated (a)avg-ESP_beta","protonated (a)avg-ESP+_beta","protonated (a)avg-ESP-_beta",
               "protonated (a)var-ESP_beta","protonated (a)var-ESP+_beta","protonated (a)var-ESP-_beta",
               "protonated (a)*PI-ESP_beta"
               ]
protonated_moments_alpha=[
               "protonated ESP-nucl_alpha","protonated NMR*delta_alpha",
               "protonated (a)*mu_alpha","protonated (a)*mu-ctb_alpha","protonated (a)tr-e*theta_alpha",
               "protonated (a)avg-ALIE_alpha","protonated (a)var-ALIE_alpha","protonated (a)max-ALIE_alpha","protonated (a)min-ALIE_alpha",
               "protonated (a)avg-LEA_alpha","protonated (a)var-LEA_alpha","protonated (a)max-LEA_alpha","protonated (a)min-LEA_alpha"
                ]
protonated_moments_beta=[
               "protonated ESP-nucl_beta","protonated NMR*delta_alpha",
               "protonated (a)*mu_beta","protonated (a)*mu-ctb_beta","protonated (a)tr-e*theta_beta",
               "protonated (a)avg-ALIE_beta","protonated (a)var-ALIE_beta","protonated (a)max-ALIE_beta","protonated (a)min-ALIE_beta",
               "protonated (a)avg-LEA_beta","protonated (a)var-LEA_beta","protonated (a)max-LEA_beta","protonated (a)min-LEA_beta"               
                ]
descriptors_alpha=protonated_charges_alpha+protonated_ESP_alpha+protonated_moments_alpha
descriptors_beta=protonated_charges_beta+protonated_ESP_beta+protonated_moments_beta
translated_symbols_alpha=[transl_symbols(s) for s in descriptors_alpha]
translated_symbols_beta=[transl_symbols(s) for s in descriptors_beta]



d_alpha=data[descriptors_alpha].corr()
rename_dict_alpha={}
for dd,tt in zip(descriptors_alpha,translated_symbols_alpha): rename_dict_alpha[dd]=tt
d_alpha=d_alpha.rename(columns=rename_dict_alpha, index=rename_dict_alpha)

d_beta=data[descriptors_beta].corr()
rename_dict_beta={}
for dd,tt in zip(descriptors_beta,translated_symbols_beta): rename_dict_beta[dd]=tt
d_beta=d_beta.rename(columns=rename_dict_beta, index=rename_dict_beta)


mask_u = np.triu(np.ones_like(d_alpha, dtype=bool))
mask_d = np.tril(np.ones_like(d_beta, dtype=bool))
d_alpha = d_alpha.mask(mask_d)
d_beta = d_beta.mask(mask_u)

heat_alpha = go.Heatmap(
    z = np.square(d_alpha),
    x = d_alpha.columns.values,
    y = d_alpha.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    orientation="h", title={"text": "r\u00b2","side":"top"},
                    y=0.5, x=0.2,len=0.25
                        #ticktext=ticktext)
                        )
)

heat_beta = go.Heatmap(
    z = np.square(d_beta),
    x = d_beta.columns.values,
    y = d_beta.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    orientation="h", title={"text": "r\u00b2","side":"top"},
                    y=0.5, x=0.7,len=0.25
                        #ticktext=ticktext)
                        )
)


title = 'Asset Correlation Matrix'

layout_a = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

layout_b = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

fig=go.Figure(data=[heat_beta], layout=layout_a)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=75,
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_prot_beta.png", width=1200, height=1200,scale=4)
fig.write_html("corr_prot_beta.html")


fig=go.Figure(data=[heat_alpha], layout=layout_b)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=75,side="top",
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",side="right",
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_prot_alpha.png", width=1200, height=1200,scale=4)
fig.write_html("corr_prot_alpha.html")



deprotonated_charges_alpha=["deprotonated PEOE_alpha",
               "deprotonated Mulliken_alpha","deprotonated Lowdin_alpha",
               "deprotonated Hirshfeld_alpha","deprotonated Voronoy_alpha","deprotonated Becke_alpha",
               "deprotonated ADCH_alpha","deprotonated CM5_alpha",#"12CM5_alpha",
               "deprotonated CHELPG_alpha","deprotonated MK_alpha","deprotonated RESP_alpha",
               "deprotonated NBO-chg_alpha"]
deprotonated_charges_beta=["deprotonated PEOE_beta",
               "deprotonated Mulliken_beta","deprotonated Lowdin_beta",
               "deprotonated Hirshfeld_beta","deprotonated Voronoy_beta","deprotonated Becke_beta",
               "deprotonated ADCH_beta","deprotonated CM5_beta",#"12CM5_beta",
               "deprotonated CHELPG_beta","deprotonated MK_beta","deprotonated RESP_beta",
               "deprotonated NBO-chg_beta"
               ]
deprotonated_ESP_alpha=[
               "deprotonated (a)Surf_alpha","deprotonated (a)Surf+_alpha","deprotonated (a)Surf-_alpha",
               "deprotonated (a)max-ESP_alpha","deprotonated (a)min-ESP_alpha",
               "deprotonated (a)avg-ESP_alpha","deprotonated (a)avg-ESP+_alpha","deprotonated (a)avg-ESP-_alpha",
               "deprotonated (a)var-ESP_alpha","deprotonated (a)var-ESP+_alpha","deprotonated (a)var-ESP-_alpha",
               "deprotonated (a)*PI-ESP_alpha"
               ]
deprotonated_ESP_beta=[
               "deprotonated (a)Surf_beta","deprotonated (a)Surf+_beta","deprotonated (a)Surf-_beta",
               "deprotonated (a)max-ESP_beta","deprotonated (a)min-ESP_beta",
               "deprotonated (a)avg-ESP_beta","deprotonated (a)avg-ESP+_beta","deprotonated (a)avg-ESP-_beta",
               "deprotonated (a)var-ESP_beta","deprotonated (a)var-ESP+_beta","deprotonated (a)var-ESP-_beta",
               "deprotonated (a)*PI-ESP_beta"
               ]
deprotonated_moments_alpha=[
               "deprotonated ESP-nucl_alpha","deprotonated NMR*delta_alpha",
               "deprotonated (a)*mu_alpha","deprotonated (a)*mu-ctb_alpha","deprotonated (a)tr-e*theta_alpha",
               "deprotonated (a)avg-ALIE_alpha","deprotonated (a)var-ALIE_alpha","deprotonated (a)max-ALIE_alpha","deprotonated (a)min-ALIE_alpha",
               "deprotonated (a)avg-LEA_alpha","deprotonated (a)var-LEA_alpha","deprotonated (a)max-LEA_alpha","deprotonated (a)min-LEA_alpha"
                ]
deprotonated_moments_beta=[
               "deprotonated ESP-nucl_beta","deprotonated NMR*delta_alpha",
               "deprotonated (a)*mu_beta","deprotonated (a)*mu-ctb_beta","deprotonated (a)tr-e*theta_beta",
               "deprotonated (a)avg-ALIE_beta","deprotonated (a)var-ALIE_beta","deprotonated (a)max-ALIE_beta","deprotonated (a)min-ALIE_beta",
               "deprotonated (a)avg-LEA_beta","deprotonated (a)var-LEA_beta","deprotonated (a)max-LEA_beta","deprotonated (a)min-LEA_beta"               
                ]
descriptors_alpha=deprotonated_charges_alpha+deprotonated_ESP_alpha+deprotonated_moments_alpha
descriptors_beta=deprotonated_charges_beta+deprotonated_ESP_beta+deprotonated_moments_beta
translated_symbols_alpha=[transl_symbols(s) for s in descriptors_alpha]
translated_symbols_beta=[transl_symbols(s) for s in descriptors_beta]



d_alpha=data[descriptors_alpha].corr()
rename_dict_alpha={}
for dd,tt in zip(descriptors_alpha,translated_symbols_alpha): rename_dict_alpha[dd]=tt
d_alpha=d_alpha.rename(columns=rename_dict_alpha, index=rename_dict_alpha)

d_beta=data[descriptors_beta].corr()
rename_dict_beta={}
for dd,tt in zip(descriptors_beta,translated_symbols_beta): rename_dict_beta[dd]=tt
d_beta=d_beta.rename(columns=rename_dict_beta, index=rename_dict_beta)


mask_u = np.triu(np.ones_like(d_alpha, dtype=bool))
mask_d = np.tril(np.ones_like(d_beta, dtype=bool))
d_alpha = d_alpha.mask(mask_d)
d_beta = d_beta.mask(mask_u)

heat_alpha = go.Heatmap(
    z = np.square(d_alpha),
    x = d_alpha.columns.values,
    y = d_alpha.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    orientation="h", title={"text": "r\u00b2","side":"top"},
                    y=0.5, x=0.2,len=0.25
                        #ticktext=ticktext)
                        )
)

heat_beta = go.Heatmap(
    z = np.square(d_beta),
    x = d_beta.columns.values,
    y = d_beta.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    orientation="h", title={"text": "r\u00b2","side":"top"},
                    y=0.5, x=0.7,len=0.25
                        #ticktext=ticktext)
                        )
)


title = 'Asset Correlation Matrix'

layout_a = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

layout_b = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

fig=go.Figure(data=[heat_beta], layout=layout_a)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=75,
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_deprot_beta.png", width=1200, height=1200,scale=4)
fig.write_html("corr_deprot_beta.html")


fig=go.Figure(data=[heat_alpha], layout=layout_b)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=75,side="top",
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",side="right",
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_deprot_alpha.png", width=1200, height=1200,scale=4)
fig.write_html("corr_deprot_alpha.html")


charges_Hrel=["PEOE*relative*H_alpha",
               "Mulliken*relative*H_alpha","Lowdin*relative*H_alpha",
               "Hirshfeld*relative*H_alpha","Voronoy*relative*H_alpha","Becke*relative*H_alpha",
               "ADCH*relative*H_alpha","CM5*relative*H_alpha",
               "CHELPG*relative*H_alpha","MK*relative*H_alpha","RESP*relative*H_alpha",
               "NBO-chg*relative*H_alpha",
                "NMR*delta*relative*H_alpha", "ESP-nucl*relative*H_alpha"
                   ]

XHbonds_rel=["IBSI*relative*H", "Mayer-BO*relative*H", 
                "WBO*relative*H", "Mulliken-BO*relative*H", 
                "FBO*relative*H", "LBO*relative*H",
                "WBO-NAO*relative*H","NBI*relative*H","NLMO-BO*relative*H",
                "FUERZA-FC*relative*H","BD*relative*H","*mu*BP-*H","*ind*mu*BP-*H",
                "diag-e*theta*BP-*H","diag-*theta*BP-*H"
                ]
descriptors_H=charges_Hrel
descriptors_B=XHbonds_rel
translated_symbols_H=[transl_symbols(s) for s in descriptors_H]
translated_symbols_B=[transl_symbols(s) for s in descriptors_B]


d_H=data[descriptors_H].corr()
d_B=data[descriptors_B].corr()

rename_dict_H={}
for dd,tt in zip(descriptors_H,translated_symbols_H): rename_dict_H[dd]=tt
d_H=d_H.rename(columns=rename_dict_H, index=rename_dict_H)

rename_dict_B={}
for dd,tt in zip(descriptors_B,translated_symbols_B): rename_dict_B[dd]=tt
d_B=d_B.rename(columns=rename_dict_B, index=rename_dict_B)


mask_u = np.tril(np.ones_like(d_H, dtype=bool))
mask_d = np.triu(np.ones_like(d_B, dtype=bool))

d_H = d_H.mask(mask_u)
d_B = d_B.mask(mask_d)


heat_H = go.Heatmap(
    z = np.square(d_H),
    x = d_H.columns.values,
    y = d_H.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    colorbar = dict(thickness=10, 
                    tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
                    orientation="h", title={"text": "r\u00b2","side":"top"},
                    y=-0.1, x=0.0,len=0.25
                        #ticktext=ticktext)
                        )
)

heat_B = go.Heatmap(
    z = np.square(d_B),
    x = d_B.columns.values,
    y = d_B.columns.values,
    zmin = 0, # Sets the lower bound of the color domain
    zmax = 1,
    xgap = 1, # Sets the horizontal gap (in pixels) between bricks
    ygap = 1,
    #colorscale = 'Blues',
    colorscale=dcolorsc,
    #colorbar = dict(thickness=10, 
    #                tickvals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1], 
    #                orientation="h", title={"text": "r\u00b2","side":"top"},
    #                y=-0.1, x=0.0,len=0.25
    #                    #ticktext=ticktext)
    #                    )
)



layout_a = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

layout_b = go.Layout(
    #title_text=title, 
    title_x=0.5, 
    width=1200, 
    height=1200,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
    plot_bgcolor="white"
)

fig=go.Figure(data=[heat_H], layout=layout_a)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=60,side="top",
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",side="right",tickangle=60,
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_H.png", width=800, height=800,scale=6)
fig.write_html("corr_H.html")


fig=go.Figure(data=[heat_B], layout=layout_b)
fig.update_xaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=60,
                 tickfont = dict(size=16,weight=800),mirror=True, title_font = dict(size=40))
fig.update_yaxes(visible=True,linewidth=0,linecolor="white",showticklabels=True,ticks="outside",tickangle=60,
                 tickfont = dict(size=15,weight=800),mirror=True, title_font = dict(size=40))

fig.write_image("corr_B.png", width=800, height=800,scale=6)
fig.write_html("corr_B.html")



