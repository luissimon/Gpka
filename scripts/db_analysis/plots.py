#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly as plt
import plotly.graph_objects as go
import sys
sys.path.append('../import')
import rdkit
import gpka_spektral_dataset

from plotly.subplots import make_subplots
from several_tautomers import several_tautomers
from feature_names import atom_features_publication_names
from feature_names import transl_symbols

data_standard=pd.read_csv("./../../models/GBDT-simple-no_micro/values_extracted-gibbs-swb97xd.25_all_1mic_with_graph_data.csv")
data_only1st=pd.read_csv("./../../models/GBDT-simple-no_micro/values_extracted-gibbs-swb97xd.25_all_with_graph_data.csv")

several_tautomers_indexes=[int(np.where(data_standard["compn"]==d)[0]) for d in several_tautomers]
data_standard=data_standard.iloc[several_tautomers_indexes]
several_tautomers_indexes=[int(np.where(data_only1st["compn"]==d)[0]) for d in several_tautomers]
data_only1st=data_only1st.iloc[several_tautomers_indexes]


data_columns=data_standard.columns

#for c in data_columns:
#    if type(data_standard[c].iloc[1])==np.float64:
#        print (c,np.sum(np.abs([data_standard[c]-data_only1st[c]])))


columns_with_changes=[c for c in data_columns if type(data_standard[c].iloc[1])==np.float64 and 
                                                  np.sum(np.abs([data_standard[c]-data_only1st[c]]))>0.01 
                                                  and "relative" not in c
                                                    and "protonated" not in c
                                                     and "deprotonated" not in c]
#print (columns_with_changes )
#print (len(columns_with_changes))
scatters,rugs,histograms=[],[],[]
text=[n+" ("+"%+d" %c+" -> "+"%+d" %(c-1)+")" for n,c in zip(data_standard['correct name'],data_standard['protonated charge'])]
titles=[transl_symbols(c) for c in columns_with_changes]
for i,c in enumerate(columns_with_changes):
    #data_standard[c]=(data_standard[c]-np.mean(data_standard[c])/np.std(data_standard[c])) # not needed: data is already standarized
    #data_only1st[c]=(data_only1st[c]-np.mean(data_only1st[c])/np.std(data_only1st[c]))
    
    
    sct=go.Scatter(y=list(data_only1st[c]),x=list(data_standard[c]),mode='markers',showlegend=False,text=text,
                                marker=dict(color="red", line=dict(width=1),showscale=False,size=4,) )
    #hist=go.Histogram(x=list( (data_standard[c]-data_only1st[c])/data_standard[c]),opacity=1.0,marker_color="red",
    #                  name=c)

    differences=(data_standard[c]-data_only1st[c])#/data_standard[c]
    rug=go.Scatter(x=list( differences),
                   y=[1.0]*len(differences),#/data_standard[c])),
                   text=text,
                   showlegend=False,mode="markers",
                   marker={"symbol":142,"color":["#D62728","#DD4477"]*len(data_standard[c]/2),"size":10},
                                        hoverinfo="text",)

    scatters.append(sct)
    rugs.append(rug)
    #histograms.append(hist)
    if (i+1)%4==0:
        fig1 = make_subplots( rows=1,cols=4,
                        #specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        subplot_titles=titles[i-3:i+1],
                        #subplot_titles=["model","residuals","errors"],
                        #row_heights=[0.6,0.35,0.05],
                        vertical_spacing=0.03,horizontal_spacing=0.05)
        fig2 = make_subplots( rows=1,cols=4,
                        #specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        subplot_titles=titles[i-3:i+1],
                        #subplot_titles=["model","residuals","errors"],
                        #row_heights=[0.6,0.35,0.05],
                        vertical_spacing=0.03,horizontal_spacing=0.05) 

        fig3 = make_subplots( rows=2,cols=4,row_heights=[0.9,0.1],
                        #specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        subplot_titles=titles[i-3:i+1],
                        #subplot_titles=["model","residuals","errors"],
                        #row_heights=[0.6,0.35,0.05],
                        vertical_spacing=0.1,horizontal_spacing=0.05) 
        
        for j,sct in enumerate(scatters[i-3:]):
            fig1.add_trace(sct,row=1,col=j%4+1)
            fig3.add_trace(sct,row=1,col=j%4+1)
        for j,rug in enumerate(rugs[i-3:]):        
            fig2.add_trace(rug,row=1,col=j%4+1)
            fig3.add_trace(rug,row=2,col=j%4+1)

        
        fig1.update_layout(height=400,width=1200,#xaxis_range=x_scale,#yaxis_range=[0,6],
                          legend={"yanchor":"top","xanchor":"right","y":0.99,"x":0.9,"font":{"size":22}})
        fig1.update_yaxes(#title_text="only 1st tautomer",title_font={'size': 22, 'weight': 1000},
                            tickfont={"size":16})
        fig1.update_xaxes(#title_text="mix of tautomers",title_font={'size': 22, 'weight': 1000},
                            tickfont={"size":16})

        fig2.update_layout(height=200,width=1200,#xaxis_range=x_scale,#yaxis_range=[0,6],
                          legend={"yanchor":"top","xanchor":"right","y":0.99,"x":0.9,"font":{"size":22}})
        fig2.update_yaxes(#title_text="only 1st tautomer",title_font={'size': 22, 'weight': 1000},
                            visible=False,tickfont={"size":16})
        fig2.update_xaxes(#title_text="mix of tautomers",title_font={'size': 22, 'weight': 1000},
                            tickfont={"size":16})


        fig3.update_layout(height=400,width=1200)
        for k in range(1,5):
            fig3.update_yaxes(row=1,col=k,tickfont={"size":16})
            fig3.update_yaxes(row=2,col=k,visible=False,range=[0,2])
            fig3.update_xaxes(row=1,col=k,tickfont={"size":16})
            fig3.update_xaxes(row=2,col=k,tickfont={"size":16})


        """
        fig1.show()
        fig1.write_html("./tautomer_scatter_plots/sct"+str(int((i+1)/4))+".html")
        fig1.write_image("./tautomer_scatter_plots/sct"+str(int((i+1)/4))+".png", width=1200, height=400,scale=1)
        
        fig2.show()
        fig2.write_html("./tautomer_scatter_plots/rug"+str(int((i+1)/4))+".html")
        fig2.write_image("./tautomer_scatter_plots/rug"+str(int((i+1)/4))+".png", width=1200, height=400,scale=1)    
        """
        i#fig3.show()
        #fig3.write_html("./tautomer_scatter_plots/sct"+str(int((i+1)/4))+".html")
        fig3.write_image("./tautomer_scatter_plots/sct"+str(int((i+1)/4))+".png", width=1200, height=400,scale=1)        
        print(i)
     

