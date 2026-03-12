import pandas as pd
import numpy as np
import plotly as plt
import plotly.graph_objects as go
import sys
sys.path.append('../import')
import rdkit
#print (rdkit.__version__)
#print (rdkit.__file__)
#import drop_compounds

CHEMBL_with_pka_db=pd.read_csv("CHEMBL_with_pka_database_with_descriptors.csv",encoding='unicode_escape')
PUBCHEM_with_pka_db=pd.read_csv("PubChem_with_pka_database_with_descriptors.csv",encoding='unicode_escape')
GPKA_db=pd.read_csv("Gpka_database_with_descriptors.csv")
SAMPL_db=pd.read_csv("SAMPL_database_with_descriptors.csv")

#remove sampl7
SAMPL_db=SAMPL_db[~SAMPL_db["name"].str.startswith("sm7")]

for db in [CHEMBL_with_pka_db,PUBCHEM_with_pka_db,GPKA_db,SAMPL_db]: 
    db.set_index("name",inplace=True)
    db.dropna(how='all', axis=1, inplace=True)
    db["PSA/MW"]=db["PSA"]/db["MW"]
#for d in drop_compounds.drop_compounds:    GPKA_db =GPKA_db[GPKA_db["compn"].str.startswith(d)==False]

CHEMBL_with_pka_small_db= CHEMBL_with_pka_db[CHEMBL_with_pka_db["MW"]<250]


import scipy

properties=["MW","n_rot_bonds","ALOGP","PSA","PSA/MW"]
name_of_properties=["Molecuar Weight","number of rotatable bonds","Calculated LOGP","Topological polar surface area",
                   "(Topological polar surface area)/MW"]
x_scales=[[0,800],[-1,16],[0,300],[-5,10],[-0.05,0.05]]
n_bins=[400,400,300,200,200]
name_of_files=["Molecuar_Weight_histogram","number_of_rotatable_bonds_histogram",
               "Calculated_LOGP_histogram","Topological_polar_surface_area_histogram",
               "Topological_polar_surface_area_histogram_relative" ]

for prop,name_of_property,x_scale,n_bins,file_name in zip(properties,name_of_properties,x_scales,n_bins,name_of_files):
    
    MW_histograms=[]
    scatters=[]
    for db,color,name in zip([GPKA_db,SAMPL_db,CHEMBL_with_pka_db,PUBCHEM_with_pka_db],
                             ["red","orange","green","blue"],["Gpka","SAMPL","CHEMBL(pKa)","PUBCHEM(pKa)"]):
        #MW_histograms.append(go.Histogram(x=db["MW"],opacity=0.75,xbins={"size":0.10},marker_color=color,legendgroup=1,legendrank=2))
        
        db[prop]=np.nan_to_num(db[prop])
        histogram=go.Histogram(x=list(db[prop]),opacity=1.0,marker_color=color,histnorm='percent',name=name,#nbinsx=n_bins,
                                autobinx=False,
                                xbins=dict(start=x_scale[0],end=x_scale[1], size=(x_scale[1]-x_scale[0])/20)
                              )
        xx=np.linspace(histogram.xbins['start'],histogram.xbins['end']+histogram.xbins['size'],100)
              
        try: 
            density = scipy.stats.gaussian_kde(db[prop])
            yy=density(xx)
            #calculate the scale
            plotbins = list(np.arange(start=histogram.xbins['start'], stop=histogram.xbins['end']+histogram.xbins['size'], step=histogram.xbins['size']))
            counts, bins = np.histogram(db[prop], bins=plotbins)
            scale_factor=100*np.max(counts)/(np.max(yy)*np.sum(counts))
            yy=yy*scale_factor
        except: 
             print ("fail")
             yy=np.zeros(len(xx))
 
        
        sct=go.Scatter(x=list(xx),y=list(yy),mode='lines',showlegend=False,hoverinfo="skip",
                       line_color=color,
                       #marker=dict(color=color, line=dict(width=1),showscale=False),
                       fill='tozeroy',
                      )
        #go.Figure(sct).show()
        MW_histograms.append(histogram)
        scatters.append(sct)
    
    fig=go.Figure(data=MW_histograms[0])
    fig.add_trace(scatters[0])
    fig.add_trace(MW_histograms[1])
    fig.add_trace(scatters[1])
    fig.add_trace(MW_histograms[2])
    fig.add_trace(scatters[2])
    fig.add_trace(MW_histograms[3])
    fig.add_trace(scatters[3])
  

    fig.update_layout(height=800,xaxis_range=x_scale,#yaxis_range=[0,6],
                      legend={"yanchor":"top","xanchor":"right","y":0.95,"x":0.92,"font":{"size":28}},
                        bargap=0.4,bargroupgap=0)
    fig.update_xaxes(title_text=name_of_property,title_font={'size': 28, 'weight': 1000},tickfont={"size":24})
    fig.update_yaxes(title_text="%",title_font={'size': 28, 'weight': 1000},tickfont={"size":24})
    
    fig.show()
    fig.write_html(file_name+".html")
    fig.write_image(file_name+".png", width=1200, height=800,scale=1)



from collections import Counter
from db_analysis_functions import shannon_entropy_scaffolds
from db_analysis_functions import normalized_shannon_entropy_scaffolds



GPKA_Murcko_Scaffolds_counts={}
CHEMBL_Murcko_Scaffolds_counts={}
PUBCHEM_Murcko_Scaffolds_counts={}
SAMPL_Murcko_scaffolds_counts={}
"""
reference_Murcko_Scaffolds_counts=Counter(list(GPKA_db["Murcko_Scaffold"])+
                                    list(CHEMBL_with_pka_db["Murcko_Scaffold"])+
                                    list(PUBCHEM_with_pka_db["Murcko_Scaffold"])+
                                    list(SAMPL_db["Murcko_Scaffold"]))
"""
ref_scaffolds=[]
for db in [PUBCHEM_with_pka_db,CHEMBL_with_pka_db]:   
    reference_Murcko_Scaffolds_counts=Counter(list(db["Murcko_Scaffold"]))
    reference_Murcko_Scaffolds_counts=dict(reference_Murcko_Scaffolds_counts.most_common())
    ref_scaffolds.append(reference_Murcko_Scaffolds_counts)

flag=True
       
for ref_scaffold,ref_name in zip(ref_scaffolds,["PUBCHEM","CHEMBL"]):                                   
    
    Murcko_histograms=[]
    for db,name,db_Murcko_scf_counts,color in zip([GPKA_db,SAMPL_db,CHEMBL_with_pka_db,PUBCHEM_with_pka_db],
                            ["GPKA              ","SAMPL            ","CHEMBL(pKa)  ","PUBCHEM(pKa)"],
                            [GPKA_Murcko_Scaffolds_counts,SAMPL_Murcko_scaffolds_counts,CHEMBL_Murcko_Scaffolds_counts,PUBCHEM_Murcko_Scaffolds_counts],
                            ["red","orange","green","blue"]):
        #Murcko_scf_counts={}
        Murcko_scf_counts=[]
        for k in ref_scaffold.keys():
            db_Murcko_scf_counts[k]=list(db["Murcko_Scaffold"]).count(k)/len(db)
            Murcko_scf_counts.append(db_Murcko_scf_counts[k])

        if flag:
            shannon_entropy=normalized_shannon_entropy_scaffolds(db["Murcko_Scaffold"])
            if shannon_entropy>=10:
                name_in_legend= name +(" scld. Shannon entropy: "+"{:.2f}".format(shannon_entropy)).rjust(10," ")
            else:    
                name_in_legend= name +(" scld. Shannon entropy: "+"{:.3f}".format(shannon_entropy)).rjust(10," ")
        else: name_in_legend=name
        x_axis_size=np.max([ len(str(t)) for t in list(ref_scaffold.keys())[0:20]  ])
        histogram=go.Bar(x=list(ref_scaffold.keys())[0:20], #list(range(0,len(Murcko_scf_counts)))[0:20],
                                        y=Murcko_scf_counts[0:20],
                                        opacity=1.0,
                                        marker_color=color,
                                        name=name_in_legend,
                                        #text= list(all_Murcko_Scaffolds_counts.keys())[0:20],hoverinfo="text"
                        )
    
            
        Murcko_histograms.append(histogram)
    
    
        
    fig2=go.Figure(data=Murcko_histograms[0])
    fig2.add_trace(Murcko_histograms[1])
    fig2.add_trace(Murcko_histograms[2])
    fig2.add_trace(Murcko_histograms[3])
    print(x_axis_size)
    #x_axis_size=0
    
    fig2.update_layout(height=800+8*x_axis_size,width=1100,#xaxis_range=x_scale,#yaxis_range=[0,6],
                        autosize=False,
                           margin=dict(
                                        l=50,
                                        r=50,
                                        b=8*x_axis_size,
                                        t=50,
                                        pad=4
                                    ),
                        #minreducedwidth=1100,
                        #minreducedheight=800,
                      legend={"yanchor":"top","xanchor":"right","y":0.85,"x":0.9,"font":{"size":28}})
    fig2.update_xaxes(title_text="",title_font={'size': 28, 'weight': 1000},tickfont={"size":16},tickangle = 60)
    fig2.update_yaxes(title_text="",title_font={'size': 28, 'weight': 1000},tickfont={"size":24})
    fig2.update_xaxes(automargin=True)
    fig2.show()
    fig2.write_html("most_fequent_Murcko_Scaffolds_wr_"+ref_name+".html")
    fig2.write_image("most_fequent_Murcko_Scaffolds"+ref_name+".png", width=1600, height=1600,scale=4)
    
    print ("20 most frequent scaffolds in CHEMBL database")
    for s in list(ref_scaffold.keys())[0:20]: print(s)
    flag=False



CHEMBL_with_pka_db=pd.read_csv("CHEMBL_with_pka_database_with_descriptors_tanimoto_with_Gpka.csv",encoding='unicode_escape')
#CHEMBL_with_pka_small_db= CHEMBL_with_pka_db[CHEMBL_with_pka_db["MW"]<300]
PUBCHEM_with_pka_db=pd.read_csv("PubChem_with_pka_database_with_descriptors_tanimoto_with_Gpka.csv",encoding='unicode_escape')
GPKA_db=pd.read_csv("Gpka_database_with_descriptors_tanimoto_with_Gpka.csv")
SAMPL_db=pd.read_csv("SAMPL_database_with_descriptors_tanimoto_with_Gpka.csv")
import scipy.stats

    
for db in [CHEMBL_with_pka_db,SAMPL_db,PUBCHEM_with_pka_db,GPKA_db]: 
    db.set_index("name",inplace=True)
    db.dropna(how='all', axis=1, inplace=True)


props=["Morgan radius 2 tanimoto similarity average","Morgan radius 2 tanimoto similarity max",
      "Morgan radius 2 tanimoto similarity median","Morgan radius 2 tanimoto similarity percentile 80",
      "Morgan radius 2 tanimoto similarity percentile 90","Morgan radius 2 tanimoto similarity percentile 90",
        "Morgan radius 3 tanimoto similarity average","Morgan radius 3 tanimoto similarity max",
      "Morgan radius 3 tanimoto similarity median","Morgan radius 3 tanimoto similarity percentile 80",
      "Morgan radius 3 tanimoto similarity percentile 90","Morgan radius 3 tanimoto similarity percentile 90",      
        "Morgan radius 4 tanimoto similarity average","Morgan radius 4 tanimoto similarity max",
      "Morgan radius 4 tanimoto similarity median","Morgan radius 4 tanimoto similarity percentile 80",
      "Morgan radius 4 tanimoto similarity percentile 90","Morgan radius 4 tanimoto similarity percentile 90", 
     "custom fingerprint tanimoto similarity average","custom fingerprint tanimoto similarity max",
      "custom fingerprint tanimoto similarity median","custom fingerprint tanimoto similarity percentile 80",
      "custom fingerprint tanimoto similarity percentile 90","custom fingerprint tanimoto similarity percentile 90",      
      ]
names_of_prop=["Morgan (R=2) fingerprint Tanimoto index (average)","Morgan (R=2) fingerprint Tanimoto index (max)",
      "Morgan (R=2) fingerprint Tanimoto index (median)","Morgan (R=2) fingerprint Tanimoto index (percentile 80)",
        "Morgan (R=2) fingerprint Tanimoto index (percentile 90)","Morgan (R=2) fingerprint Tanimoto index (percentile 95)",
       "Morgan (R=3) fingerprint Tanimoto index (average)","Morgan (R=3) fingerprint Tanimoto index (max)",
      "Morgan (R=3) fingerprint Tanimoto index (median)","Morgan (R=3) fingerprint Tanimoto index (percentile 80)",
        "Morgan (R=3) fingerprint Tanimoto index (percentile 90)","Morgan (R=3) fingerprint Tanimoto index (percentile 95)",
       "Morgan (R=4) fingerprint Tanimoto index (average)","Morgan (R=4) fingerprint Tanimoto index (max)",
      "Morgan (R=4) fingerprint Tanimoto index (median)","Morgan (R=4) fingerprint Tanimoto index (percentile 80)",
        "Morgan (R=4) fingerprint Tanimoto index (percentile 90)","Morgan (R=4) fingerprint Tanimoto index (percentile 95)",
       "functional groups fingerprint Tanimoto index (average)","functional groups fingerprint Tanimoto index (max)",
      "functional groups fingerprint Tanimoto index (median)","functional groups fingerprint Tanimoto index (percentile 80)",
        "functional groups fingerprint Tanimoto index (percentile 90)","functional groups fingerprint Tanimoto index (percentile 95)",      
      ]
file_names=["MorganR2average","MorganR2max","MorganR2median","MorganR2perc80","MorganR2perc90","MorganR2perc95",
            "MorganR3average","MorganR3max","MorganR3median","MorganR3perc80","MorganR3perc90","MorganR3perc95",
            "MorganR4average","MorganR4max","MorganR4median","MorganR4perc80","MorganR4perc90","MorganR4perc95",
            "functgrpsaverage","functgrpsmax","functgrpsmedian","functgrpsperc80","functgrpsperc90","functgrpsperc95",
            ]
scales=[[0.0,0.3],[0,1.2],[0,0.3],[0,0.3],[0,0.8],[0,1.2]]*4
scales=[[0.0,s] for s in [0.22,1.1,0.22,0.34,0.34,0.34,0.16,1.1,0.16,0.24,0.28,0.28,0.16,1.1,0.16,0.22,0.26,0.26,0.42,1.1,0.42,0.22,0.26,0.26]]

legend_position=[[0.4,0.95]]*len(props)
for prop,name_of_property,scale,legend_position,file_name in zip(props,names_of_prop,scales,legend_position,file_names):
    MW_histograms=[]
    scatters=[]
    for db,color,name in zip([GPKA_db,SAMPL_db,CHEMBL_with_pka_db,PUBCHEM_with_pka_db],
                             ["red","orange","green","blue"],
                             ["GpKa","SAMPL","CHEMBL(pKa)","PUBCHEM(pKa)"]):
            #MW_histograms.append(go.Histogram(x=db["MW"],opacity=0.75,xbins={"size":0.10},marker_color=color,legendgroup=1,legendrank=2))
            histogram=go.Histogram(x=list(db[prop]),opacity=1.0,marker_color=color,histnorm='percent',name=name,
                                              autobinx=False,#nbinsx=20
                                                 xbins=dict(start=scale[0],end=scale[1], size=(scale[1]-scale[0])/20))

            xx=np.linspace(histogram.xbins['start'],histogram.xbins['end']+histogram.xbins['size'],100)
            try: 
                density = scipy.stats.gaussian_kde(db[prop])
                yy=density(xx)
                #calculate the scale
                plotbins = list(np.arange(start=histogram.xbins['start'], stop=histogram.xbins['end']+histogram.xbins['size'], step=histogram.xbins['size']))
                counts, bins = np.histogram(db[prop], bins=plotbins)
                scale_factor=100*np.max(counts)/(np.max(yy)*np.sum(counts))
                yy=yy*scale_factor
            except: yy=np.zeros(len(xx))
            sct=go.Scatter(x=list(xx),y=list(yy),mode='lines',showlegend=False,hoverinfo="skip",
                           line_color=color,
                           #marker=dict(color=color, line=dict(width=1),showscale=False),
                           fill='tozeroy',
                          )
            #go.Figure(sct).show()
            MW_histograms.append(histogram)
            scatters.append(sct)
    
        
    fig=go.Figure(data=MW_histograms[0])
    fig.add_trace(scatters[0])
    fig.add_trace(MW_histograms[1])
    fig.add_trace(scatters[1])
    fig.add_trace(MW_histograms[2])
    fig.add_trace(scatters[2])
    fig.add_trace(MW_histograms[3])
    fig.add_trace(scatters[3])
       
    fig.update_layout(height=800,xaxis_range=scale,#yaxis_range=[0,6],
                          legend={"yanchor":"top","xanchor":"right","y":legend_position[1],"x":legend_position[0],"font":{"size":34}},
                      bargap=0.6,bargroupgap=0.0)
    fig.update_xaxes(title_text=name_of_property,title_font={'size': 32, 'weight': 1000},tickfont={"size":28})
    fig.update_yaxes(title_text="%",title_font={'size': 36, 'weight': 1000},tickfont={"size":24})
        
    fig.show()
    fig.write_html(file_name+".html")
    fig.write_image(file_name+".png", width=1200, height=800,scale=1)


CHEMBL_with_pka_db=pd.read_csv("CHEMBL_with_pka_database_with_descriptors_tversky_with_Gpka.csv",encoding='unicode_escape')
#CH,line_shape='spline'EMBL_with_pka_small_db= CHEMBL_with_pka_db[CHEMBL_with_pka_db["MW"]<300]
PUBCHEM_with_pka_db=pd.read_csv("PubChem_with_pka_database_with_descriptors_tversky_with_Gpka.csv",encoding='unicode_escape')
GPKA_db=pd.read_csv("Gpka_database_with_descriptors_tversky_with_Gpka.csv")
SAMPL_db=pd.read_csv("SAMPL_database_with_descriptors_tversky_with_Gpka.csv")

import scipy.stats
    
for db in [CHEMBL_with_pka_db,SAMPL_db,PUBCHEM_with_pka_db,GPKA_db]: 
    db.set_index("name",inplace=True)
    db.dropna(how='all', axis=1, inplace=True)

MW_histograms=[]
props=["Morgan radius 2 tversky similarity average","Morgan radius 2 tversky similarity max",
      "Morgan radius 2 tversky similarity median","Morgan radius 2 tversky similarity percentile 80",
      "Morgan radius 2 tversky similarity percentile 90","Morgan radius 2 tversky similarity percentile 90",
        "Morgan radius 3 tversky similarity average","Morgan radius 3 tversky similarity max",
      "Morgan radius 3 tversky similarity median","Morgan radius 3 tversky similarity percentile 80",
      "Morgan radius 3 tversky similarity percentile 90","Morgan radius 3 tversky similarity percentile 90",      
        "Morgan radius 4 tversky similarity average","Morgan radius 4 tversky similarity max",
      "Morgan radius 4 tversky similarity median","Morgan radius 4 tversky similarity percentile 80",
      "Morgan radius 4 tversky similarity percentile 90","Morgan radius 4 tversky similarity percentile 90", 
     "custom fingerprint tversky similarity average","custom fingerprint tversky similarity max",
      "custom fingerprint tversky similarity median","custom fingerprint tversky similarity percentile 80",
      "custom fingerprint tversky similarity percentile 90","custom fingerprint tversky similarity percentile 90",      
      ]
names_of_prop=["Morgan (R=2) fingerprint Tversky(0,1) index (average)","Morgan (R=2) fingerprint Tversky(0,1) index (max)",
      "Morgan (R=2) fingerprint Tversky(0,1) index (median)","Morgan (R=2) fingerprint Tversky(0,1) index (percentile 80)",
        "Morgan (R=2) fingerprint Tversky(0,1) index (percentile 90)","Morgan (R=2) fingerprint Tversky(0,1) index (percentile 95)",
       "Morgan (R=3) fingerprint Tversky(0,1) index (average)","Morgan (R=3) fingerprint Tversky(0,1) index (max)",
      "Morgan (R=3) fingerprint Tversky(0,1) index (median)","Morgan (R=3) fingerprint Tversky(0,1) index (percentile 80)",
        "Morgan (R=3) fingerprint Tversky(0,1) index (percentile 90)","Morgan (R=3) fingerprint Tversky(0,1) index (percentile 95)",
       "Morgan (R=4) fingerprint Tversky(0,1) index (average)","Morgan (R=4) fingerprint Tversky(0,1) index (max)",
      "Morgan (R=4) fingerprint Tversky(0,1) index (median)","Morgan (R=4) fingerprint Tversky(0,1) index (percentile 80)",
        "Morgan (R=4) fingerprint Tversky(0,1) index (percentile 90)","Morgan (R=4) fingerprint Tversky(0,1) index (percentile 95)",
       "functional groups fingerprint Tversky(0,1) index (average)","functional groups fingerprint Tversky(0,1) index (max)",
      "functional groups fingerprint Tversky(0,1) index (median)","functional groups fingerprint Tversky(0,1) index (percentile 80)",
        "functional groups fingerprint Tversky(0,1) index (percentile 90)","functional groups fingerprint Tversky(0,1) index (percentile 95)",      
      ]
file_names=["MorganR2tverskyaverage","MorganR2tverskymax","MorganR2tverskymedian","MorganR2tverskyperc80","MorganR2tverskyperc90","MorganR2tverskyperc95",
            "MorganR3tverskyaverage","MorganR3tverskymax","MorganR3tverskymedian","MorganR3tverskyperc80","MorganR3tverskyperc90","MorganR3tverskyperc95",
            "MorganR4tverskyaverage","MorganR4tverskymax","MorganR4tverskymedian","MorganR4tverskyperc80","MorganR4tverskyperc90","MorganR4tverskyperc95",
            "functgrp_tverskysaverage","functgrp_tverskysmax","functgrp_tverskysmedian","functgrp_tverskysperc80","functgrp_tverskysperc90","functgrp_tverskysperc95",
            ]
scales=[[0.0,0.3],[0,1.2],[0,0.3],[0,0.3],[0,0.8],[0,1.2]]*4
scales=[[0.0,s] for s in [0.45,1.1,0.45,0.6,0.65,0.65,0.4,1.1,0.4,0.5,0.6,0.6,0.4,1.1,0.4,0.45,0.5,0.5,1.0,1.0,1.0,0.5,0.6,0.6]]
#scales=[[0,1]]*24
legend_position=[[0.4,0.95]]*len(props)
for prop,name_of_property,scale,legend_position,file_name in zip(props,names_of_prop,scales,legend_position,file_names):
    MW_histograms=[]
    scatters=[]
    for db,color,name in zip([GPKA_db,SAMPL_db,CHEMBL_with_pka_db,PUBCHEM_with_pka_db],
                             ["red","orange","green","blue"],
                             ["GpKa","SAMPL","CHEMBL(pKa)","PUBCHEM(pKa)"]):
            #MW_histograms.append(go.Histogram(x=db["MW"],opacity=0.75,xbins={"size":0.10},marker_color=color,legendgroup=1,legendrank=2))
#
            
            histogram=go.Histogram(x=list(db[prop]),opacity=1.0,marker_color=color,histnorm='percent',name=name,
                                              autobinx=False,#nbinsx=20
                                                 xbins=dict(start=scale[0],end=scale[1], size=(scale[1]-scale[0])/20))

            xx=np.linspace(histogram.xbins['start'],histogram.xbins['end']+histogram.xbins['size'],100)
            try: 
                density = scipy.stats.gaussian_kde(db[prop])
                yy=density(xx)
                #calculate the scale
                plotbins = list(np.arange(start=histogram.xbins['start'], stop=histogram.xbins['end']+histogram.xbins['size'], step=histogram.xbins['size']))
                counts, bins = np.histogram(db[prop], bins=plotbins)
                scale_factor=100*np.max(counts)/(np.max(yy)*np.sum(counts))
                yy=yy*scale_factor
            except: yy=np.zeros(len(xx))
            sct=go.Scatter(x=list(xx),y=list(yy),mode='lines',showlegend=False,hoverinfo="skip",
                           line_color=color,
                           #marker=dict(color=color, line=dict(width=1),showscale=False),
                           fill='tozeroy',
                          )
            #go.Figure(sct).show()
            MW_histograms.append(histogram)
            scatters.append(sct)
                                      
    #kde=ff.create_distplot(distplot_data,names_of_prop,show_rug=False)
    #kde.show()            
        
    fig=go.Figure(data=MW_histograms[0])
    fig.add_trace(scatters[0])
    fig.add_trace(MW_histograms[1])
    fig.add_trace(scatters[1])
    fig.add_trace(MW_histograms[2])
    fig.add_trace(scatters[2])
    fig.add_trace(MW_histograms[3])
    fig.add_trace(scatters[3])
       
    fig.update_layout(height=800,xaxis_range=scale,#yaxis_range=[0,6],
                          legend={"yanchor":"top","xanchor":"right","y":legend_position[1],"x":legend_position[0],"font":{"size":34}},
                      bargap=0.6,bargroupgap=0.0)
    fig.update_xaxes(title_text=name_of_property,title_font={'size': 32, 'weight': 1000},tickfont={"size":28})
    fig.update_yaxes(title_text="%",title_font={'size': 36, 'weight': 1000},tickfont={"size":24})
        
    fig.show()
    fig.write_html(file_name+".html")
    fig.write_image(file_name+".png", width=1200, height=800,scale=1)



from routes import extracted_data_route
from drop_compounds import drop_compounds
import copy
lot="swb97xd"
csv_file=extracted_data_route+"values_extracted-gibbs-"+lot+".25.csv" 
data=pd.read_csv(csv_file,low_memory=True)
data.dropna(axis=0)

outliers=data[data["compn"].isin( drop_compounds)]
data=data[~data["compn"].isin(drop_compounds)]

#data.set_index("compn",inplace=True)
#outliers.set_index("compn",inplace=True)

from sklearn.linear_model import LinearRegression
test_attributes='deltaG'
X=np.c_[data['deltaG']]
Y=np.array(data["pKa"].copy())
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
pka_prediction=lin_reg.predict(X)


fig3=go.Figure(data=go.Scatter(
                               y=list(pka_prediction),x=list(Y),mode='markers',showlegend=False,hoverinfo="skip",
                                marker=dict(color="lightgray", line=dict(width=1),showscale=False) 
                              )) 


outliers_X,outliers_Y= np.c_[outliers[test_attributes]],np.array(outliers["pKa"].copy())
outliers_pka_prediction=lin_reg.predict(outliers_X)
fig3.add_trace(go.Scatter( 
                            y=list(outliers_pka_prediction),x=list(outliers_Y),mode='markers',text=outliers["compn"],showlegend=False,
                                marker=dict(color="red", line=dict(width=2),showscale=False) 
                              )) 
fig3.update_layout(height=1100,width=1100,#xaxis_range=x_scale,#yaxis_range=[0,6],
                  legend={"yanchor":"top","xanchor":"right","y":0.99,"x":0.9,"font":{"size":22}})
fig3.update_xaxes(title_text="published pKa",title_font={'size': 22, 'weight': 1000},tickfont={"size":16})
fig3.update_yaxes(title_text="predicted pKa",title_font={'size': 22, 'weight': 1000},tickfont={"size":16})
fig3.show()
fig3.write_html("dropped_compounds.html")
fig3.write_image("dropped_compounds.png", width=1600, height=1600,scale=4)




from routes import extracted_data_route
from drop_compounds import drop_compounds
import copy
lot="swb97xd"
csv_file=extracted_data_route+"values_extracted-gibbs-swb97xd.25.csv" 
json_graph_file=extracted_data_route+"molecular_graphs-gibbs-swb97xd.25.json"

import gpka_spektral_dataset
labels=pd.read_csv(csv_file)
dataset=gpka_spektral_dataset.gpka_spektral_dataset(json_graph_file,csv_file=csv_file,equilibrium_keys=[],
                                linear_equilibrium_keys=[],label_key="pKa")



number_of_microeqs=[]
number_of_microeqs2=[]
number_of_nodes=[]
for G in dataset.graphs: 
    matrix=G.e.transpose(2,1,0)[0]
    n=dataset.number_of_microeqs(matrix)
    number_of_microeqs.append(n)
    number_of_microeqs2.append(np.sum(G.mask))
    number_of_nodes.append(len(matrix)/n)




number_of_microeqs_hist=go.Histogram(x=number_of_microeqs,opacity=1.0,marker_color="red",name="number of microequilibriums",
                                     #nbinsx=80
                                    )
number_of_nodes_hist=go.Histogram(x=number_of_nodes,opacity=1.0,marker_color="red",name="number of microequilibriums",
                                     #nbinsx=80
                                    )
number_of_total_nodes=list(np.array(number_of_microeqs)*np.array(number_of_nodes))

number_of_total_nodes_hist=go.Histogram(x=number_of_total_nodes,opacity=1.0,marker_color="red",name="number of microequilibriums",
                                     nbinsx=60
                                    )
hists=[number_of_microeqs_hist,number_of_nodes_hist,number_of_total_nodes_hist]
texts=["number of microequilibriums","number of nodes in each microequilibrium","total number of nodes"]
file_names=["number_of_microequilibriums","number_of_nodes","total_number_of_nodes"]
for hist,text in zip(hists,texts):

    fig=go.Figure(data=hist)
    fig.update_layout(height=800,#xaxis_range=scale,#yaxis_range=[0,6],
                              legend={"yanchor":"top","xanchor":"right","y":0.90,"x":0.3,"font":{"size":42}},bargap=0.4,bargroupgap=0)

    fig.update_xaxes(title_text=text,title_font={'size': 22, 'weight': 1000},tickfont={"size":16})
    fig.update_yaxes(title_text="number of entries",title_font={'size': 22, 'weight': 1000},tickfont={"size":16})
    fig.show()
    fig.write_html(file_name+".html")
    fig.write_image(file_name+".png", width=1600, height=1600,scale=4)
    




from scipy import spatial

number_of_microeqs=[]
number_of_microeqs2=[]
number_of_nodes=[]
for G in dataset.graphs: 
    matrix=G.e.transpose(2,1,0)[0]
    n=dataset.number_of_microeqs(matrix)
    number_of_microeqs.append(n)
    number_of_microeqs2.append(np.sum(G.mask))
    number_of_nodes.append(len(matrix)/n)


np.set_printoptions(precision=2,linewidth=175)
errors=0
#several_tautomers=[G.name for G in dataset.graphs if np.sum(G.mask)>1]
#print(several_tautomers)
#print(dataset.atom_feature_keys)
        
#print (len(several_tautomers)) 
counter=0
differences_summary={}
for G in dataset.graphs:

    if np.sum(G.mask)>1:
        counter+=1
        
        mask_pieces=np.array_split(G.mask,np.sum(G.mask))
        weighted_mask_pieces=np.array_split(G.weighted_mask,np.sum(G.mask))
        weights=[np.sum(w) for w in weighted_mask_pieces]
        
        max_weight_pieces_index=np.argmax([np.sum(v) for v in weighted_mask_pieces])
        new_mask=list(np.concatenate([mp*(int(i==max_weight_pieces_index)) for i,mp in enumerate(mask_pieces)]))
        scale_factor=1.0/np.sum(weighted_mask_pieces[max_weight_pieces_index])
        new_mask=[x*scale_factor for x in new_mask]

        
        GxT=G.x.transpose(1,0)
        at_prop_normal,at_prop_1st=[],[]
        for index in range (0,len(GxT)):
            prop_normal=np.array(GxT[index]).dot(G.weighted_mask)
            prop_only1st=np.array(GxT[index]).dot(new_mask)       
            at_prop_normal.append(prop_normal)
            at_prop_1st.append(prop_only1st)

        cos_sim=spatial.distance.cosine(at_prop_normal,at_prop_1st)

        #print(np.array(at_prop_normal[0:16]))
        #print(np.array(at_prop_1st[0:16]))
        #print ("scale factor",scale_factor)
        #print ("cos sim ",cos_sim)                                       
        #print ("----")
        differences_summary[G.name]=[scale_factor,cos_sim]
        if counter>50000: 
            print (counter)
            para_esta_locura
#min_cos_similarity=np.min([v[1] for v in differences_summary.values()])
differences_summary_sorted_by_scale_factor=dict(sorted(differences_summary.items(), key=lambda item: item[1][0]))
differences_summary_sorted_by_cos_sim=dict(sorted(differences_summary.items(), key=lambda item: item[1][1]))      
print("based on scaling factor")
print(list(differences_summary_sorted_by_scale_factor.keys())[0:100])

#for k in list(differences_summary_sorted_by_scale_factor.keys())[0:10]:
#    v=differences_summary_sorted_by_scale_factor[k]
#    print (k," with scale factor: ",v[0]," and cos similarity: ",v[1])
print("based on cos similarity")
print(list(differences_summary_sorted_by_cos_sim.keys())[0:100])

#for k in list(differences_summary_sorted_by_cos_sim.keys())[0:10]:
#    v=differences_summary_sorted_by_scale_factor[k]
#    print (k," with scale factor: ",v[0]," and cos similarity: ",v[1])


    

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
scatters,histograms=[],[]
text=[n+" ("+"%+d" %c+" -> "+"%+d" %(c-1)+")" for n,c in zip(data_standard['correct name'],data_standard['protonated charge'])]
for i,c in enumerate(columns_with_changes[0:80]):
    
    sct=go.Scatter(y=list(data_only1st[c]),x=list(data_standard[c]),mode='markers',showlegend=False,text=text,
                                marker=dict(color="red", line=dict(width=1),showscale=False) )
    #hist=go.Histogram(x=list( (data_standard[c]-data_only1st[c])/data_standard[c]),opacity=1.0,marker_color="red",
    #                  name=c)

    scatters.append(sct)
    #histograms.append(hist)
    if (i+1)%4==0:
        fig1 = make_subplots( rows=1,cols=4,
                        #specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        subplot_titles=[transl_symbols(c) for c in columns_with_changes],
                        #subplot_titles=["model","residuals","errors"],
                        #row_heights=[0.6,0.35,0.05],
                        vertical_spacing=0.03,horizontal_spacing=0.05)
        for i,sct in enumerate(scatters[i-3:]):
            fig1.add_trace(sct,row=1,col=i%4+1)
        fig1.update_layout(height=400,width=1200,#xaxis_range=x_scale,#yaxis_range=[0,6],
                          legend={"yanchor":"top","xanchor":"right","y":0.99,"x":0.9,"font":{"size":22}})
        fig1.update_yaxes(#title_text="only 1st tautomer",title_font={'size': 22, 'weight': 1000},
                            tickfont={"size":16})
        fig1.update_xaxes(#title_text="mix of tautomers",title_font={'size': 22, 'weight': 1000},
                            tickfont={"size":16})
        fig1.show() 
        
    



""" 
fig1 = make_subplots( rows=20,cols=4,
                        #specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        subplot_titles=[transl_symbols(c) for c in columns_with_changes],
                        #subplot_titles=["model","residuals","errors"],
                        #row_heights=[0.6,0.35,0.05],
                        vertical_spacing=0.03,horizontal_spacing=0.05)
for i,sct in enumerate(scatters):
    fig1.add_trace(sct,row=int(i/4)+1,col=i%4+1)


#fig1.update_annotations(font=font)
#fig1.update_layout(legend=dict(font=dict(size=18),yanchor="top",xanchor="right",y=0.2,x=0.2))
    

fig1.update_layout(height=8200,width=1200,#xaxis_range=x_scale,#yaxis_range=[0,6],
                  legend={"yanchor":"top","xanchor":"right","y":0.99,"x":0.9,"font":{"size":22}})
fig1.update_yaxes(#title_text="only 1st tautomer",title_font={'size': 22, 'weight': 1000},
                    tickfont={"size":16})
fig1.update_xaxes(#title_text="mix of tautomers",title_font={'size': 22, 'weight': 1000},
                    tickfont={"size":16})
fig1.show()
                
fig2 = make_subplots( rows=7,cols=4,
                        #specs=[  [{'rowspan': 3}, {"b":0.1}]  , [None, {}] ,[None, {}]  ],
                        subplot_titles=columns_with_changes,
                        #subplot_titles=["model","residuals","errors"],
                        #row_heights=[0.6,0.35,0.05],
                        vertical_spacing=0.03,horizontal_spacing=0.05)
for i,hist in enumerate(histograms):
    fig2.add_trace(hist,row=int(i/4)+1,col=i%4+1)


#fig1.update_annotations(font=font)
#fig2.update_layout(legend=dict(font=dict(size=18),yanchor="top",xanchor="right",y=0.2,x=0.2))
    
for i in range(1,5):
    for j in range(1,8):
        fig2.update_yaxes(range=[0,6],row=j,col=i)
    
fig2.update_layout(height=2100,width=1200,#xaxis_range=x_scale,
                   #yaxis_range=[0,6],
                  #legend={"yanchor":"top","xanchor":"right","y":0.99,"x":0.9,"font":{"size":22}}
                  )
fig2.update_yaxes(#title_text="only 1st tautomer",title_font={'size': 22, 'weight': 1000},
                    tickfont={"size":16})
fig2.update_xaxes(#title_text="mix of tautomers",title_font={'size': 22, 'weight': 1000},
                    tickfont={"size":16})
fig2.show()    
"""



