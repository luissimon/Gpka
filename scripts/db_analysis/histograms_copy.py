import pandas as pd
import numpy as np
import plotly as plt
import plotly.graph_objects as go
import sys
sys.path.append('../import')
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


import rdkit
print (rdkit.__version__)
print (rdkit.__file__)


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


        

        

several_tautomers= ['123triazole_cation->neut',
 '123triazole_neut->an',
 '13cyclohexanedione_neut->an',
 '17phenanthroline_cation->neut',
 '1hydroxyacridine_cation->neut',
 '1hydroxyacridine_neut->an',
 '1methylxanthine_an->2an',
 '1phenylurazole_neut->an',
 '24dimethylimidazole_cation->neut',
 '26dimethylpyrazine_cation->neut',
 '26dimethylpyrimidine_cation->neut',
 '2aminopyrazine_cation->neut',
 '2dimethylaminopyrazine_cation->neut',
 '4methyl34dihydroquinazoline_cation->neut',
 '2methylaminopyrazine_cation->neut',
 '2methylpyrazine_cation->neut',
 '2methylquinazoline_cation->neut',
 '2ph5phimidazole_cation->neut',
 '2ph5phimidazole_neut->an',
 '3methylxanthine_neut->an',
 '3mercaptopyridine_cation->neut',
 '3mercaptopyridine_neut->an',
 '3methoxypyridazine_cation->neut',
 '3methylthiopyridazine_cation->neut',
 '4-aza-benzotriazoleoxime_neut->an',
 '4hydroxyisoquinoline_cation->neut',
 '4hydroxypteridine_cation->neut',
 '4methylpyrimidine_cation->neut',
 '4methylquinazoline_cation->neut',
 '5aminosalicylic_cation->neut',
 '5aminosalicylic_neut->an',
 '5methylquinazoline_cation->neut',
 '5nitro110phenanthroline_cation->neut',
 '5phenylimidazole_cation->neut',
 '6hydroxyisoquinoline_cation->neut',
 '6methylquinazoline_cation->neut',
 '7-aza-benzotriazoleoxime_neut->an',
 '7hydroxyquinoline_cation->neut',
 '7hydroxyquinoline_neut->an',
 '7methylquinazoline_cation->neut',
 '7methylxanthine_neut->an',
 '7methylxanthine_an->2an',
 '8methylquinazoline_cation->neut',
 '8oxoguanine_neut->an',
 'acetohydroxamic_neut->an',
 'adenine_neut->an',
 'adenineNoxyde_neut->an',
 'aminotetrazole_neut->an',
 'aminotetrazole_cation->neut',
 'benzenehydroxamic_neut->an',
 'benzotriazole_cation->neut',
 'bromotetrazole_neut->an',
 'chlorotetrazole_neut->an',
 'cinnoline_cation->neut',
 'dimethylguanidine_cation->neut',
 'dopamine_cation->neut',
 'dopamine_neut->an',
 'guanine_an->2an',
 'histidine_cation->neut',
 'histidine_neut->an',
 'hypoxanthine_cation->neut',
 'hypoxanthine_neut->an',
 'lumacine_neut->an',
 'lysine_cation->neut',
 'lysine_neut->an',
 'malic_neut->an',
 'malic_an->2an',
 'mercaptopurine_neut->an',
 'mesaconic_neut->an',
 'mesaconic_an->2an',
 'methyltetrazole_neut->an',
 'ornitine_cation->neut',
 'ornitine_neut->an',
 'picolinic_cation->neut',
 'picolinic_neut->an',
 'purine_neut->an',
 'quinazoline_cation->neut',
 'tetrazole_neut->an',
 'theophilline_neut->an',
 'thymine_neut->an',
 'tolylhydroquinone_neut->an',
 'tolylhydroquinone_an->2an',
 'uracil_neut->an',
 'urazole_neut->an',
 'uric_neut->an',
 'uric_an->2an',
 '2367tetrahydroimiazo12apyrimidin51hone_cation->neut',
 '4methylthiocinnoline_cation->neut',
 '1methylthionaphtalazine_cation->neut',
 '2methoxyquinoxaline_cation->neut',
 '4carbamoylimidazole_cation->neut',
 '3methylpyrazole_cation->neut',
 '2methoxypyrazine_cation->neut',
 '2methylthiopyrazine_cation->neut',
 '2carbamoylpyrazine_cation->neut',
 '2amino4methylpyrimidine_cation->neut',
 '4amino6chloropyrimidine_cation->neut',
 'benzenepentacarboxylic_neut->an',
 '1235benzenetetracarboxylic_3an->4an',
 '1235benzenetetracarboxylic_neut->an',
 '1234benzenetetracarboxylic_3an->4an',
 '123benzenetricarboxylic_2an->3an',
 '123benzenetricarboxylic_an->2an',
 '123benzenetricarboxylic_neut->an',
 'nphenylimidazolin2imine_cation->neut',
 '4hydroxypyrimidine_neut->an',
 'hydroxytropolone_neut->an',
 'hydroxytropolone_an->2an',
 'histamine_2cation->cation',
 'histamine_cation->neut',
 '235trimethylpyrrole_cation->neut',
 'benzotriazole_neut->an',
 '8mercaptoquinoline_cation->neut',
 '8mercaptoquinoline_neut->an',
 '3mercaptoisoquinoline_cation->neut',
 '3mercaptoisoquinoline_neut->an',
 '5hydroxycinnoline_cation->neut',
 '5hydroxycinnoline_neut->an',
 '6hydroxycinnoline_cation->neut',
 '6hydroxycinnoline_neut->an',
 '7hydroxycinnoline_cation->neut',
 '7hydroxycinnoline_neut->an',
 '8hydroxycinnoline_cation->neut',
 '5hydroxyquinoxaline_cation->neut',
 '6hydroxyquinoxaline_cation->neut',
 '1hydroxyphenazine_cation->neut',
 '2hydroxyphenazine_cation->neut',
 '2hydroxyphenazine_neut->an',
 '1methoxyphtalazine_cation->neut',
 '2hydroxy135triazaindene_neut->an',
 '2mercapto135triazaindene_neut->an',
 '6chloro2hydroxy135triazaindene_neut->an',
 '6chloro2hydroxy135triazaindene_an->2an',
 '6chloro135triazaindene_neut->an',
 '4chloro135triazaindene_neut->an',
 '6chloro2mercapto135triazaindene_neut->an',
 '6chloro2mercapto135triazaindene_an->2an',
 '26dichloro135triazaindene_cation->neut',
 '26dichloro135triazaindene_neut->an',
 'Nmethyl5amino2oxopyridine_cation->neut',
 '36dihydroxypyridazine_neut->an',
 '2methylthio135triazaindene_neut->an',
 'epinastine_cation->neut',
 'estazolam_cation->neut',
 '5fluorouracil_neut->an',
 '5fluorouracil_an->2an',
 'histaminemonoiodo_2cation->cation',
 'histaminemonoiodo_neut->an',
 'histaminediiodo_2cation->cation',
 'histaminediiodo_neut->an',
 '7hydroxyphenanthridine_cation->neut',
 '7hydroxyphenanthridine_neut->an',
 '2methyl34dihydroquinazoline_cation->neut',
 'quinmerac_neut->an',
 '1methylurazole_neut->an',
 '1methylxanthine_neut->an',
 'adenine_cation->neut',
 'adenineNoxyde_cation->neut',
 'allopurinol_neut->an',
 'guanine_cation->neut',
 'isoguanine_cation->neut',
 'isoguanine_neut->an',
 'purine_cation->neut',
 '6chloro135triazaindene_cation->neut',
 '2methylthioquinoxaline_cation->neut',
 '4hydroxypyrimidine_cation->neut',
 '6chloro2methylthio135triazaindene_cation->neut',
 '6chloro2methylthio135triazaindene_neut->an',
 '2methylthio135triazaindene_cation->neut',
 '18dihydroxyadenine_cation->neut',
 '18dihydroxyadenine_neut->an',
 '18dihydroxyadenine_an->2an',
 '669trimethyladenine_2cation->cation',
 '669trimethyladenine_cation->neut',
 '9methylisoguanine_cation->neut',
 '9methylisoguanine_neut->an',
 '9methyladenine_2cation->cation',
 '9methyladenine_cation->neut',
 'isocytosine_cation->neut',
 'isocytosine_neut->an',
 '56dimethyl2thiouracil_neut->an',
 '6methyl2thiouracil_neut->an',
 '5methyl2thiouracil_neut->an',
 '4dimethylamino6methoxypyrimidine_cation->neut',
 '4dimethylamino6methylthiopyrimidine_cation->neut',
 '4dimethylamino6mercaptopyrimidine_cation->neut',
 '4dimethylamino6hydroxypyrimidine_cation->neut',
 '4amino6methylthiopyrimidine_cation->neut',
 '4amino6mercaptopyrimidine_cation->neut',
 '4amino6hydroxypyrimidine_cation->neut',
 '2thiouracil_neut->an',
 '2dimethylamino4hydroxypyrimidine_cation->neut',
 '2dimethylamino4hydroxypyrimidine_neut->an',
 '2amino4mercaptopyrimidine_cation->neut',
 '2amino4mercaptopyrimidine_neut->an',
 '4mercaptopyrimidine_neut->an',
 'azobenzene34diol_neut->an',
 'azobenzene34diol_an->2an',
 '4nitroazobenzene34diol_an->2an',
 '3methoxy6ptolyl45dihydropyridazine_cation->neut',
 '16dihydro1methyl4methylamino6thiopyrimidine_cation->neut',
 '16dihydro1methyl4amino6thiopyrimidine_cation->neut',
 '4hydroxy6methylaminopyrimidine_cation->neut',
 '4methoxy6methylaminopyrimidine_cation->neut',
 '3phenyl4benzoyl5isoxazolone_neut->an',
 '3methyl4benzoyl5isoxazolone_neut->an',
 'pyrazole3carboxylic_neut->an',
 'quinazolinehydrate_cation->neut',
 '4methyl2methylthioquinazoline_cation->neut',
 '2amino1456tetrahydropyrimidine_cation->neut',
 '22466pentamethyl1256tetrahydropyrimidine_cation->neut',
 '2amino5bromo4tertbutylpyrimidine_cation->neut',
 '2amino5bromo4methylpyrimidine_cation->neut',
 '2amino5bromo4hydroxypyrimidine_cation->neut',
 '2amino5bromo4hydroxypyrimidine_neut->an',
 '2amino5chloro4hydroxypyrimidine_cation->neut',
 '2amino5chloro4hydroxypyrimidine_neut->an',
 '2amino5fluoro4hydroxypyrimidine_cation->neut',
 '2amino5fluoro4hydroxypyrimidine_neut->an',
 '2amino5cyano4methylpyrimidine_cation->neut',
 '4acetamidopyrimidine_cation->neut',
 '4amino6chloro2hydroxypyrimidine_cation->neut',
 '4amino6chloro2hydroxypyrimidine_neut->an',
 '5bromo2mercaptopyrimidine_neut->an',
 '5bromo4hydroxypyrimidine_cation->neut',
 '5bromo4hydroxypyrimidine_neut->an',
 '46dihydroxypyrimidine_cation->neut',
 '46dimercaptopyrimidine_neut->an',
 'acridine4carboxylic_neut->an',
 '5amino23dimethylpyrazine_cation->neut',
 '2amino3dimethylaminopyrazine_2cation->cation',
 '2amino3dimethylaminopyrazine_cation->neut',
 'isoxazolidine5carboxylic_neut->an',
 'isoxazolidine5carboxylic_cation->neut',
 'pyridine26dimethyl35dicarboxylic_cation->neut',
 'pyridine26dimethyl35dicarboxylic_neut->an',
 'pyridine35dicarboxylic_cation->neut',
 'pyridine35dicarboxylic_neut->an',
 'pyridine35dicarboxylic_an->2an',
 'chelidamic_an->2an',
 'pyridine345tricarboxylic_2an->3an',
 'xanthopterin_cation->neut',
 'xanthopterin_neut->an',
 'xanthopterin_an->2an',
 '8oxypterin_cation->neut',
 '8oxypterin_neut->an',
 '2amino56dihydro6thioxopteridin43Hone_cation->neut',
 '2amino56dihydro6thioxopteridin43Hone_neut->an',
 '2amino56dihydro6thioxopteridin43Hone_an->2an',
 '2amino7sulfanylidene38dihydropteridin4one_cation->neut',
 '2amino7sulfanylidene38dihydropteridin4one_neut->an',
 '2amino7sulfanylidene38dihydropteridin4one_an->2an',
 'isoxanthopterin_cation->neut',
 'isoxanthopterin_neut->an',
 'isoxanthopterin_an->2an',
 '3hydroxy6dimethylaminopyridazine_cation->neut',
 '4carboxylictropolone_an->2an',
 '4carboxylictropolone_neut->an',
 '5carboxylictropolone_neut->an',
 '5carboxylictropolone_an->2an',
 '21iminotetramethylene2imidazoline_cation->neut',
 'cyclopenta25dien12dcarbonitrile_neut->an',
 'cyclopenta25dien25dcarbonitrile_neut->an',
 'nmethylhydrazine_cation->neut',
 'nndimethylhydrazine_cation->neut',
 'nnnptrimethylhydrazine_cation->neut',
 'isopropylhidrazine_cation->neut',
 'pyrrolidin1amine_cation->neut',
 'piperidine1amine_cation->neut',
 '1aminohomopiperidine_cation->neut',
 'tolylhydrazine_cation->neut',
 '1458tetraazaphenanthrene_cation->neut',
 '46dihydroxypyrimidine_neut->an',
 'acetylacetone_neut->an',
 'malonaldehyde_neut->an',
 '2hydroxypyrimidin5ylphosphonic_neut->an',
 '2aminopyrimidin5ylphosphonic_neut->an',
 '5methyl13cyclohexanedione_neut->an',
 '2methyl13cyclohexanedione_neut->an',
 '2acetyl13cyclohexanedione_neut->an',
 '2acetyl55dimethyl13cyclohexanedione_neut->an',
 '5furany2yl13cyclohexanedione_neut->an',
 '5pehnyl13cyclohexanedione_neut->an',
 '5chloropehnyl13cyclohexanedione_neut->an',
 '5fluoropehnyl13cyclohexanedione_neut->an',
 '44dimethyl13cyclohexanedione_neut->an',
 '2hydroxy4methylpyrimidin5ylphosphonic_neut->an',
 '2hydroxy4methylpyrimidin5ylphosphonic_an->2an',
 '2hydroxy4methylpyrimidin5ylphosphonic_2an->3an',
 'diacetylguanazline_neut->an',
 '6methyl2amino4hydroxypteridine_neut->an',
 '2nitropropan2one_neut->an',
 '3methyl24pentanedione_neut->an',
 '2amino367trimethylpteridin43Hone_cation->neut',
 '2methylamino67trimethylpteridin43Hone_cation->neut',
 '2methylamino367trimethylpteridin43Hone_cation->neut',
 '2amino67diphenylpteridin43Hone_cation->neut',
 '2amino67diphenylpteridin43Hone_neut->an',
 '2dimethylamino67diphenylpteridin43Hone_cation->neut',
 '2dimethylamino67diphenylpteridin43Hone_neut->an',
 '2thioxo23dihydropteridine471H8Hdione_neut->an',
 '2thioxo23dihydropteridine471H8Hdione_an->2an',
 '2thioxo23dihydropteridine471H8Hdione_2an->3an',
 '1methyl2thioxo23dihydropteridine471H8Hdione_neut->an',
 'pteridine473H8Hdione_neut->an',
 'pteridine473H8Hdione_an->2an',
 '2thioxo1235tetrahydropteridine46dione_neut->an',
 '2thioxo1235tetrahydropteridine46dione_an->2an',
 '2thioxo1235tetrahydropteridine46dione_2an->3an',
 '1methyl2thioxo1235tetrahydropteridine46dione_neut->an',
 '1methyl2thioxo1235tetrahydropteridine46dione_an->2an',
 '46dioxo3456tetrahydropteridine2sulfonic_an->2an',
 '46dioxo3456tetrahydropteridine2sulfonic_2an->3an',
 '3methyl2thioxo2358tetrahydropteridine4671Htrione_neut->an',
 '3methyl2thioxo2358tetrahydropteridine4671Htrione_an->2an',
 '3methyl2thioxo2358tetrahydropteridine4671Htrione_2an->3an',
 '2thioxo2358tetrahydropteridine4671Htrione_neut->an',
 '2thioxo2358tetrahydropteridine4671Htrione_2an->3an',
 '2methylthio58dihydropteridine4673Htrione_neut->an',
 '2methylthio58dihydropteridine4673Htrione_an->2an',
 '2methylthio58dihydropteridine4673Htrione_2an->3an',
 '58dihydropteridine4673Htrione_an->2an',
 '58dihydropteridine4673Htrione_2an->3an',
 '1methylisoxanthopterin_cation->neut',
 '1methylisoxanthopterin_neut->an',
 '16dimethylisoxanthopterin_cation->neut',
 '7mercapto13dimethyl6thioxo68adihydropteridine241H3Hdione_neut->an',
 '2ethylquinazoline_cation->neut',
 '4ethylquinazoline_cation->neut',
 '4isopropylquinazoline_cation->neut',
 '8methoxyquinazoline_cation->neut',
 '7methoxyquinazoline_cation->neut',
 '6methoxyquinazoline_cation->neut',
 '5methoxyquinazoline_cation->neut',
 '135triazanaphthalene_cation->neut',
 '2methylquinazolinehydrate_cation->neut',
 '5methylquinazolinehydrate_cation->neut',
 '7methylquinazolinehydrate_cation->neut',
 '8methylquinazolinehydrate_cation->neut',
 '26dimethylquinazolinehydrate_cation->neut',
 '4carbamoylquinazolinehydrate_cation->neut',
 '4carbamoylquinazolinehydrate_neut->an',
 '4carbamoyl2methylquinazolinehydrate_cation->neut',
 '4carbamoyl2methylquinazolinehydrate_neut->an',
 '4carbamoyl2methyl6fluoroquinazolinehydrate_cation->neut',
 '4carbamoyl2methyl6fluoroquinazolinehydrate_neut->an',
 '5methoxyquinazolinehydrate_cation->neut',
 '6methoxyquinazolinehydrate_cation->neut',
 '7methoxyquinazolinehydrate_cation->neut',
 '8methoxyquinazolinehydrate_cation->neut',
 '7ethylthioqunazolinehydrate_cation->neut',
 '2ethylquinazolinehydrate_cation->neut',
 '2isopropylquinazolinehydrate_cation->neut',
 '5chloroquinazolinehydrate_cation->neut',
 '6chloroquinazolinehydrate_cation->neut',
 '7chloroquinazolinehydrate_cation->neut',
 '8chloroquinazolinehydrate_cation->neut',
 '5fluoroquinazolinehydrate_cation->neut',
 '6fluoroquinazolinehydrate_cation->neut',
 '7fluoroquinazolinehydrate_cation->neut',
 '6fluor2methylquinazolinehydrate_cation->neut',
 '5nitroquinazolinehydrate_cation->neut',
 '6nitroquinazolinehydrate_cation->neut',
 '7nitroquinazolinehydrate_cation->neut',
 '8nitroquinazolinehydrate_cation->neut',
 '26dihcloropyrimidinone_neut->an',
 '46dichloropyrimidinone_neut->an',
 '4chloro6methylaminopyrimidin2one_cation->neut',
 '4chloro6methylaminopyrimidin2one_neut->an',
 '7thiol24diaminopteridine_cation->neut',
 '7thiol24diaminopteridine_neut->an',
 '4amino2dimethylaminopteridine7thiol_cation->neut',
 'rubalcb4_neut->an',
 'rubalcb5_neut->an',
 'rubalcb5_an->2an',
 'rubalcb6_neut->an',
 'rubalcb6_an->2an',
 '7dimethylamino13dimethylpteridine241H3Hdione_cation->neut',
 '7amino13dimethylpteridine241H3Hdione_cation->neut',
 '7methylamino13dimethylpteridine241H3Hdione_cation->neut',
 '2amino6methyl4oxo34dihydropteridine7carboxylic_cation->neut',
 '2amino6methyl4oxo34dihydropteridine7carboxylic_neut->an',
 '2amino6methyl4oxo34dihydropteridine7carboxylic_an->2an',
 'methyl2amino6methyl4oxo34dihydropteridine7carboxylic_cation->neut',
 'methyl2amino6methyl4oxo34dihydropteridine7carboxylic_neut->an',
 '2amino7methyl4oxo34dihydropteridine6carboxylic_cation->neut',
 '2amino7methyl4oxo34dihydropteridine6carboxylic_neut->an',
 '2amino7methyl4oxo34dihydropteridine6carboxylic_an->2an',
 '78dihydro10H13thiazolo23bpteridin10one_cation->neut',
 '89dihydro7H11H13thiazino23bpteridin11one_cation->neut',
 '23dimethyl78dihydro10H13thiazolo23bpteridin10one_cation->neut',
 '23dimethyl89dihydro5H13thiazolo32apteridin5one_cation->neut',
 'phenylbiguanide_cation->neut',
 'oxypurinol_neut->an',
 '8nitroxanthine_neut->an',
 '8nitroxanthine_an->2an',
 '1methyl8nitroxanthine_neut->an',
 '3methyl8nitroxanthine_neut->an',
 '7methyl8nitroxanthine_neut->an',
 '13dimethyl8nitroxanthine_neut->an',
 '8diazoxanthine_neut->an',
 '8aminoxanthine_neut->an',
 '8aminoxanthine_an->2an',
 '1methyl8aminoxanthine_neut->an',
 '1methyl8aminoxanthine_an->2an',
 '3methyl8aminoxanthine_neut->an',
 '3methyl8aminoxanthine_an->2an',
 '4fluoro1hbenzimidazole_cation->neut',
 '4fluoro1hbenzimidazole_neut->an',
 '45difluoro1hbenzimidazole_cation->neut',
 '45difluoro1hbenzimidazole_neut->an',
 '46difluoro1hbenzimidazole_cation->neut',
 '46difluoro1hbenzimidazole_neut->an',
 '456difluoro1hbenzimidazole_cation->neut',
 '456difluoro1hbenzimidazole_neut->an',
 '457trifluoro1hbenzimidazole_cation->neut',
 '457trifluoro1hbenzimidazole_neut->an',
 '5fluoro1hbenzimidazole_cation->neut',
 '1pyridin2ylpiperazine_cation->neut',
 'methylbiguanide_cation->neut',
 'dimethylbiguanide_cation->neut',
 '4tolylbiguanide_cation->neut',
 '45imidazoledicarboxylic_an->2an',
 '5amino124triazole_cation->neut',
 '5amino124triazole_neut->an',
 '123triazole4carboxylic_neut->an',
 '123triazole4carboxylic_an->2an',
 'urocanic_neut->an',
 '222tripyridine_2cation->cation',
 '222tripyridine_cation->neut',
 '4aminotropolone_cation->neut',
 '3aminotropolone_cation->neut',
 '67dimethyl45diazaindan_cation->neut',
 '6methyl45diazaindan_cation->neut',
 '45diazaindan_cation->neut',
 '5guanidinio1Htetrazole_cation->neut',
 '4br7oh3methyltropolone_neut->an',
 '3br2oh6methyltropolone_neut->an',
 '2br7oh4methyltropolone_neut->an',
 'benzoyleneurea_neut->an',
 '3methoxytropone_neut->an',
 '3methyltropolone_neut->an',
 '4methyltropolone_neut->an',
 '5nitrobenzimidazole_neut->an',
 '46dinitro1hindazol7ol_neut->an',
 'trans1me12cyclpropdicarbxylic_neut->an',
 'trans1me12cyclpropdicarbxylic_an->2an',
 'trans1ph12cyclpropdicarbxylic_neut->an',
 'trans1ph12cyclpropdicarbxylic_an->2an',
 'trans1me2cl12cyclpropdicarbxylic_neut->an',
 'trans1me2cl12cyclpropdicarbxylic_an->2an',
 'trans1ph2cl12cyclpropdicarbxylic_neut->an',
 'trans1ph2cl12cyclpropdicarbxylic_an->2an',
 'cis1me12cyclpropdicarbxylic_neut->an',
 'cis1me12cyclpropdicarbxylic_an->2an',
 '1s2r3rcis13dimecyclpropdicarbxylic_neut->an',
 '1s2r3rcis13dimecyclpropdicarbxylic_an->2an',
 '1s2r3scis13dimecyclpropdicarbxylic_neut->an',
 '1s2r3scis13dimecyclpropdicarbxylic_an->2an',
 '1r2r3strans13dimecyclpropdicarbxylic_neut->an',
 '1r2r3strans13dimecyclpropdicarbxylic_an->2an',
 '1r2r3rtrans13dimecyclpropdicarbxylic_neut->an',
 '1r2r3rtrans13dimecyclpropdicarbxylic_an->2an',
 '1s2s3rtrans1cl3mecyclpropdicarbxylic_neut->an',
 '1s2s3rtrans1cl3mecyclpropdicarbxylic_an->2an',
 'jr9650003339n23_neut->an',
 'jr9650003339n23_an->2an',
 'jr9650003339n24_neut->an',
 'jr9650003339n24_an->2an',
 '2r3r5oxooxolane23dicarbx_neut->an',
 '2r3r5oxooxolane23dicarbx_an->2an',
 '2s3r5oxooxolane23dicarbx_neut->an',
 '2s3r5oxooxolane23dicarbx_an->2an',
 '4hydroxytropolone_neut->an',
 '4hydroxytropolone_an->2an',
 'furan24dicarboxylic_neut->an',
 'furan24dicarboxylic_an->2an',
 'Nhydroxypyridine4thione_cation->neut',
 'Nhydroxypyridine4thione_neut->an',
 '2dimeaminopyridinenoxide_cation->neut',
 '5ureidoimidazole4carboxylic_cation->neut',
 '5ureidoimidazole4carboxylic_neut->an',
 '5amino1cyclohex4imidazolecarboxylic_cation->neut',
 '5amino1cyclohex4imidazolecarboxylic_neut->an',
 '4imidazolecarboxylic_cation->neut',
 '4imidazolecarboxylic_neut->an',
 '5methylpyrazole3carboxylic_neut->an',
 '46dihd1me124triazine351H2Hdione_neut->an',
 '6me135triazine241H3Hdione_neut->an',
 '56dihydro6azauracil_neut->an',
 '5hydroxy6azauracil_neut->an',
 '5amino1H123triazole4carbonitrile_neut->an',
 '24diaminonaphthalen1ol_2cation->cation',
 '24diaminonaphthalen1ol_cation->neut',
 '2aminomethylpyridine_2cation->cation',
 '2aminomethylpyridine_cation->neut',
 '4mesilamidepyridine_cation->neut',
 '4mesilamidepyridine_neut->an',
 '23pbipyridine_2cation->cation',
 '23pbipyridine_cation->neut',
 '24pbipyridine_2cation->cation',
 '24pbipyridine_cation->neut',
 '34pbipyridine_2cation->cation',
 '34pbipyridine_cation->neut',
 'nndimethylethyldenediamine_2cation->cation',
 'nndimethylethyldenediamine_cation->neut',
 '12propanediamine_2cation->cation',
 '12propanediamine_cation->neut',
 '12diamino2methylpropane_2cation->cation',
 '12diamino2methylpropane_cation->neut',
 '24diaminoresorcinol_2cation->cation',
 '24diaminoresorcinol_cation->neut',
 '24diaminoresorcinol_neut->an',
 '24diaminoresorcinol_an->2an',
 '24diaminophenol_2cation->cation',
 '24diaminophenol_cation->neut',
 'benzene124triamine_cation->neut',
 'benzene124triamine_2cation->cation',
 '246triaminoresorcin_3cation->2cation',
 '246triaminoresorcin_2cation->cation',
 '246triaminoresorcin_cation->neut',
 '24dihydroxyaniline_neut->an',
 '24dihydroxyaniline_an->2an',
 '5sulpho7br8ohquinoline_neut->an',
 '5sulpho7br8ohquinoline_an->2an',
 'quinoline6carboxylic_neut->an',
 '4aminobenzimidazole_cation->neut',
 '5aminobenzimidazole_cation->neut',
 '1methyl7aminobenzimidazole_cation->neut',
 '2methyl5aminobenzimidazole_cation->neut',
 '4methylbenzimidazole_cation->neut',
 '5nitro2mebenzimidazole_cation->neut',
 '5nitro2mebenzimidazole_neut->an',
 '4nitrobenzimidazole_cation->neut',
 '4amino1hindazole_cation->neut',
 '5amino1hindazole_cation->neut',
 '7amino1hindazole_cation->neut',
 '16naphthyridine_cation->neut',
 '17naphtyridine_cation->neut',
 '8oh16naphthyridine_cation->neut',
 '2aminopurine_2cation->cation',
 '2aminopurine_cation->neut',
 '2aminopurine_neut->an',
 '8aminopurine_cation->neut',
 '8aminopurine_neut->an',
 '68biscf37hpurin2amine_cation->neut',
 '68biscf37hpurin2amine_neut->an',
 '2am8phpurine_cation>neut',
 '2am8phpurine_neut->an',
 '6cf39hpurin2amine_cation->neut',
 '6cf39hpurin2amine_neut->an',
 '6cyanopurine_cation->neut',
 '6cyanopurine_neut->an',
 '26diaminopurine_cation->neut',
 '26diaminopurine_neut->an',
 '8cf326diaminopurine_cation->neut',
 '8cf326diaminopurine_neut->an',
 '27dimehypoxanthine_cation->neut',
 'purine8carboxylic_cation->neut',
 'purine8carboxylic_neut->an',
 'purine8carboxylic_an->2an',
 '16dih8oh1me6oxopurine_neut->an',
 '1methyladenine_cation->neut',
 '1methyl2hydroxypurine_cation->neut',
 '1methyl2hydroxypurine_neut->an',
 '26dihydroxypurine_neut->an',
 '26dihydroxypurine_an->2an',
 '28dihydroxypurine_cation->neut',
 '28dihydroxypurine_neut->an',
 '28dihydroxypurine_an->2an',
 '9methyl28dihydroxypurine_neut->an',
 '9methyl28dihydroxypurine_an->2an',
 '2dimeampurine_cation->neut',
 '2dimeampurine_neut->an',
 '6dimeampurine_cation->neut',
 '6dimeampurine_neut->an',
 '8dimeampurine_cation->neut',
 '8dimeampurine_neut->an',
 '6dimeamcarbamoylpurine_neut->an',
 '6oh2cf3purine_neut->an',
 '6oh2cf3purine_an->2an',
 '2methoxypurine_cation->neut',
 '2methoxypurine_neut->an',
 '6methoxypurine_cation->neut',
 '6methoxypurine_neut->an',
 '8methoxypurine_cation->neut',
 '8methoxypurine_neut->an',
 '6methylpurine_cation->neut',
 '6methylpurine_neut->an',
 '8methylpurine_cation->neut',
 '8methylpurine_neut->an',
 '9methylpurine_cation->neut',
 '2methylthiopurine_cation->neut',
 '2methylthiopurine_neut->an',
 '6methylthiopurine_cation->neut',
 '6methylthiopurine_neut->an',
 '8methylthiopurine_cation->neut',
 '8methylthiopurine_neut->an',
 '8phenylpurine_cation->neut',
 '8phenylpurine_neut->an',
 '268triaminopurine_2cation->cation',
 '268triaminopurine_cation->neut',
 '268triaminopurine_neut->an',
 '6cf3purine_neut->an',
 '8cf3purine_cation->neut',
 '8cf3purine_neut->an',
 '1hpyrazolo34dpyrimidine_cation->neut',
 '7amidopyrazolo43dpyrimidine_cation->neut',
 '7amidopyrazolo43dpyrimidine_neut->an',
 '3me7ampyrazolo43dpyrimidine_cation->neut',
 '4ampyrazolo34dpyrimidine_cation->neut',
 '16dimepyrazolo34dpyrimidin4am_cation->neut',
 '6mepyrazolo34dpyrimidin4am_cation->neut',
 '3mepyrazolo34dpyrimidin4am_cation->neut',
 '3mepyrazolo34dpyrimidin4am_neut->an',
 '6me1phpyrazolo34dpyrimidin4am_cation->neut',
 '1phpyrazolo34dpyrimidin4am_cation->neut',
 '1hpyrazolo34dpyrimidin46diam_cation->neut',
 '1ph1hpyrazolo34dpyrimidin46diam_cation->neut',
 'pyrazolo34dpyrimidin4dimeam_cation->neut',
 '1mepyrazolo34dpyrimidin4dimeam_cation->neut',
 '4meo1mepyrazolo34dpyrimidine_cation->neut',
 '4mespyrazolo34dpyrimidine_cation->neut',
 '7amthiazolo54dpyrimidin_cation->neut',
 '1hnaphth12dimidazole_cation->neut',
 '1hnaphth12dimidazole_neut->an',
 '2chl6789tetrah1hnapht12dimidazole_cation->neut',
 '2dimeam6789tetrah1hnapht12dimidazole_cation->neut',
 '6789tetrah1hnapht12dimidazole_cation->neut',
 '6789tetrah1hnapht12dimidazole_neut->an',
 '15phenanthroline_2cation->cation',
 '15phenanthroline_cation->neut',
 '234567me18phenanthroline_cation->neut',
 '4br110phenanthroline_cation->neut',
 '4cl110phenanthroline_cation->neut',
 '5cl110phenanthroline_cation->neut',
 '4cl2me110phenanthroline_cation->neut',
 '24dime110phenanthroline_cation->neut',
 '34dime110phenanthroline_cation->neut',
 '37dime110phenanthroline_cation->neut',
 '46dime110phenanthroline_cation->neut',
 '2me110phenanthroline_cation->neut',
 '3me110phenanthroline_cation->neut',
 '5me110phenanthroline_cation->neut',
 '4et110phenanthroline_cation->neut',
 '3et110phenanthroline_cation->neut',
 '4oh110phenanthroline_cation->neut',
 '3467tetrame110phenanthroline_cation->neut',
 '3468tetrame110phenanthroline_cation->neut',
 '346trime110phenanthroline_cation->neut',
 '347trime110phenanthroline_cation->neut',
 '356trime110phenanthroline_cation->neut',
 '357trime110phenanthroline_cation->neut',
 '358trime110phenanthroline_cation->neut',
 'benzoc18naphthyridine_cation->neut',
 '7amnme13thiazolo54dpyrimidin_cation->neut',
 '7am5methiazolo54dpyrimidine_cation->neut',
 '4amnme1hpyrazolo34dpyrimidin_cation->neut',
 '16dime4meampyrazolo34dpyrimidine_cation->neut',
 '6meo1me1hpyrazolo34dpyrimidin4amine_cation->neut',
 '6cl4am1me1hpyrazolo34dpyrimidin_cation->neut',
 '8meampurine_cation->neut',
 '8meampurine_neut->an',
 '2me6meampurine_cation->neut',
 '6nmecarbamoylpurine_cation->neut',
 '6nmecarbamoylpurine_neut->an',
 '7methylpurine_cation->neut',
 '4azabenzimidazole_cation->neut',
 '4azabenzimidazole_neut->an',
 '1methylphthalazine_cation->neut',
 '4am6chlnmepyrimidin_cation->neut',
 '4dimeam6chlpyrimidin_cation->neut',
 '1me4meampyrazolo34dpyrimidine_cation->neut',
 '6me4ampyrazolo34dpyrimidine_cation->neut',
 '4oh135triazanaphthalene_neut->an',
 '8oh145triazanaphthalene_cation->neut',
 '5oh146triazanaphthalene_cation->neut',
 'pteridine4acetamide_cation->neut',
 'pteridine7amino_cation->neut',
 'pteridine47dime2amino_cation->neut',
 '46diaminopteridine_cation->neut',
 'pteridine467triamino_cation->neut',
 '67dihydroxypteridine_neut->an',
 '67dihydroxypteridine_an->2an',
 '34dih367trime4oxopteridine_cation->neut',
 '7ohpteridine_cation->neut',
 '7methoxypteridine_cation->neut',
 '3aminocinnoline_cation->neut',
 '5aminocinnoline_cation->neut',
 '3amino6chlorocinnoline_cation->neut',
 '3amino7chlorocinnoline_cation->neut',
 '8oh4mecinnoline_cation->neut',
 '5amquinazoline_cation->neut',
 '7amquinazoline_cation->neut',
 '34dih4tbuquinazoline_cation->neut',
 '4ohquinazoline_neut->an',
 '34dihquinazoline_cation->neut',
 '4me34dihquinazoline_cation->neut',
 '2oh78dih6mepteridine_neut->an',
 'ethylhydrazine_cation->neut',
 '34dioh1naphtylamine_neut->an',
 '14dioh2naphtylamine_neut->an',
 '14dioh2naphtylamine_an->2an',
 'picolinohydrazide_cation->neut',
 'chlorcyclizine_cation->neut',
 '56diam2meampyrimidin43hone_cation->neut',
 '456triamn4mepyrimidine_cation->neut',
 '46dioh2dimeampyrimidine_cation->neut',
 '6me1234tetrahydroquinozaline_cation->neut',
 '2methylquinoxaline_cation->neut',
 '26dihydroxypyridine_cation->neut',
 '26dihydroxypyridine_neut->an',
 '4tbu2ampyrimidine_cation->neut',
 '5carbamoyl4me2ampyrimidine_cation->neut',
 '4tbu2ohpyrimidine_cation->neut',
 '4tbu2ohpyrimidine_neut->an',
 '4aminomethylimidazole_2cation->cation',
 '4aminomethylimidazole_cation->neut',
 'imidazole4methanol_cation->neut',
 '4methylimidazole_cation->neut',
 '42pyridilimidazole_2cation->cation',
 '42pyridilimidazole_cation->neut',
 '2phenolimidazolin2yl_neut->an',
 'nicotone_cation->neut',
 'cyclizine_cation->neut',
 '3am56dimepyrazine2carboxylic_neut->an',
 '2mercaptopyrazine_cation->neut',
 '3me1ph1Hpyrazol5ol_cation->neut',
 '3me1ph1Hpyrazol5ol_neut->an',
 '3ethylpyridazine_cation->neut',
 '3aminopyridazine_cation->neut',
 '3amino6mepyridazine_cation->neut',
 '3et6mepyridazine_cation->neut',
 '4et3mepyridazine_cation->neut',
 '3methylpyridazine_cation->neut',
 '4methylpyridazine_cation->neut',
 '346trimethylpyridazine_cation->neut',
 '6cl2nme24diampyrimidin_cation->neut',
 '4am6methoxpyrimidine_cation->neut',
 '5am4mepyrimidine_cation->neut',
 '4nme46diampyrimidine_cation->neut',
 'n4methyl5nitropyrimidine46diam_cation->neut',
 '5bromouracyl_neut->an',
 '6hdyroxypyrimidine4carbox_neut->an',
 '6hdyroxypyrimidine4carbox_an->2an',
 '25diamino46dihydroxypyrimidine_neut->an',
 '6methylS24pyrimidinediamine_cation->neut',
 '5hydroxypyrimidine43hone_cation->neut',
 '5hydroxypyrimidine43hone_neut->an',
 '5hydroxypyrimidine43hone_an->2an',
 '6methyluracil_neut->an',
 'n4n4n6trime26diampyrimidine_cation->neut',
 '5methoxypyrimidin4ol_cation->neut',
 '5methoxypyrimidin4ol_neut->an',
 '2methoxy6mepyrimidin41hone_neut->an',
 '4mepyrimidin2ol_cation->neut',
 '4mepyrimidin2ol_neut->an',
 '4hydroxy6methylpyrimidine_cation->neut',
 '4hydroxy6methylpyrimidine_neut->an',
 '4oh2meampyrimidine_cation->neut',
 '4oh2meampyrimidine_neut->an',
 '4me2mercaptopyrimidine_cation->neut',
 '4me2mercaptopyrimidine_neut->an',
 '6me4mercaptopyrimidine_cation->neut',
 '6me4mercaptopyrimidine_neut->an',
 '4me2methiopyrimidine_cation->neut',
 '245triaminepyrimidine_2cation->cation',
 '5ph124thiadiazol3amine_cation->neut',
 '3ph124thiadiazol5amine_cation->neut',
 '356trimethyltriazine_cation->neut',
 'isoquinoline3hydroxy_cation->neut',
 'isoquinoline3hydroxy_neut->an',
 '24diam138triazanaphtalene_cation->neut',
 '57diamthiazolo54dpyrimidine_cation->neut',
 '26diam8azapurine_cation->neut',
 '26diam8azapurine_neut->an',
 '24diam10hpyrimido54b14benzothiazine_cation->neut',
 '4carboxam26diampyrimidine_cation->neut',
 '24diam5carboxpyrimidine_cation->neut',
 '6cl24diampyrimidine_cation->neut',
 '6methoxy24diampyrimidine_cation->neut',
 '5nitro24diampyrimidine_cation->neut',
 '6phenoxi24diampyrimidine_cation->neut',
 '5ph6cf324diampyrimidine_cation->neut',
 '6meam24diam135triazine_cation->neut',
 '6phenox24diam135triazine_cation->neut',
 '4oh6me2am135triazine_cation->neut',
 '4oh6ph2am135triazine_cation->neut',
 '4oh6ph2am135triazine_neut->an',
 '37dimeindolizine_cation->neut',
 '3meindolizine_cation->neut',
 '6meindolizine_cation->neut',
 '3me124triazole_cation->neut',
 '3me124triazole_neut->an',
 '3ph124triazole_cation->neut',
 '3ph124triazole_neut->an',
 '3et124triazole_cation->neut',
 '3et124triazole_neut->an',
 '35diam124triazole_cation->neut',
 '4bromo35mepyrazole_cation->neut',
 '4chloro3ethylpyrazole_cation->neut',
 '4chloro35methylpyrazole_cation->neut',
 '4chloro35phenylpyrazole_cation->neut',
 '354dimetpyrazole_cation->neut',
 '354dime53phepyrazole_cation->neut',
 '35ethylpyrazole_cation->neut',
 '35me53phepyrazole_cation->neut',
 '4me3phepyrazole_cation->neut',
 '35phenylpyrazole_cation->neut',
 '4567tetrahindazole_cation->neut',
 '3me4567tetrahindazole_cation->neut',
 '4formyl123triazole_neut->an',
 '5clbenzimidazole_cation->neut',
 '5clbenzimidazole_neut->an',
 '4clbenzimidazole_cation->neut',
 '5brbenzimidazole_cation->neut',
 '5brbenzimidazole_neut->an',
 '4brbenzimidazole_cation->neut',
 '5mebenzimidazole_cation->neut',
 '5mebenzimidazole_neut->an',
 '5cf3benzimidazole_cation->neut',
 '5cf3benzimidazole_neut->an',
 '2am123treiazolo5p4p54pyrimidine_neut->an',
 '8azaguanine_cation->neut',
 '8azaguanine_neut->an',
 '2am6me123triazolo5p4p54pyrimidine_cation->neut',
 '2am6me123triazolo5p4p54pyrimidine_neut->an',
 '16dih123triazolo5p4p54pyrimidine_cation->neut',
 '16dih123triazolo5p4p54pyrimidine_neut->an',
 '2me16dih123triazolo5p4p54pyrimidine_cation->neut',
 '2me16dih123triazolo5p4p54pyrimidine_neut->an',
 '2oh123triazolo5p4p54pyrimidine_an->2an',
 '2oh6me123triazolo5p4p54pyrimidine_cation->neut',
 '2oh6me123triazolo5p4p54pyrimidine_neut->an',
 '2oh6me123triazolo5p4p54pyrimidine_an>2an',
 '6me123triazolo5p4p54pyrimidine_cation->neut',
 '6me123triazolo5p4p54pyrimidine_neut->an',
 '6me2mes123triazolo5p4p54pyrimidine_cation->neut',
 '6me2mes123triazolo5p4p54pyrimidine_neut->an',
 '2mes123triazolo5p4p54pyrimidine_cation->neut',
 '2mes123triazolo5p4p54pyrimidine_neut->an',
 '8azapurine_neut->an',
 '2clpurine_cation->neut',
 '2clpurine_neut->an',
 '6clpurine_cation->neut',
 '6clpurine_neut->an',
 '8clpurine_cation->neut',
 '8clpurine_neut->an',
 '26diclpurine_neut->an',
 '268triclpurine_neut->an',
 '5me7meamthiazolo45dpyrimidine_cation->neut',
 '2oh8mespurine_cation->neut',
 '2oh8mespurine_neut->an',
 '2oh8mespurine_an->2an',
 '2oh6mepurine_cation->neut',
 '2oh6mepurine_neut->an',
 '2oh6mepurine_an->2an',
 '2am8cf3purine_cation->neut',
 '2am8cf3purine_neut->an',
 '2am8mespurine_cation->neut',
 '2am8mespurine_neut->an',
 '8mesulfonylpurine_cation->neut',
 '8mesulfonylpurine_neut->an',
 '2oh5ampyridine_cation->neut',
 '24dioh5ampyrimidine_neut->an',
 '6meam4s2me5ampyrimidine_cation->neut',
 '2me5am4meam6mespyrimidine_cation->neut',
 '69diimepurine_cation->neut',
 '3me6dimeampurine_cation->neut',
 '2piperidiniopurine_cation->neut',
 '2piperidiniopurine_neut->an',
 '6piperidiniopurine_cation->neut',
 '6piperidiniopurine_neut->an',
 '23diam6clpyridine_2cation->cation',
 '23diam6clpyridine_cation->neut',
 '3cyclopropyl1hpyrazole_cation->neut',
 '4am5mescarbimiazole_cation->neut',
 '4am5mescarbimiazole_neut->an',
 '1pme6mes123azol5p4p45pyrmdine_cation->neut',
 '4me5carboxyimidazole_cation->neut',
 '4me5carboxyimidazole_neut->an',
 '4me5carboxythiazole_neut->an',
 '22pyridilimidazole_2cation->cation',
 '22pyridilimidazole_cation->neut',
 '3me4oxo2meimimidazolidine_cation->neut',
 '4am1phpyrazole_cation->neut',
 '3tbu1hpyrazole_cation->neut',
 '4br35me53phpyrazole_cation->neut',
 '4br3phpyrazole_cation->neut',
 '15dime3oxo23dihpyrazole_cation->neut',
 '15dime3oxo23dihpyrazole_neut->an',
 '34dime1ph5oxo25dihpyrazole_cation->neut',
 '34dime1ph5oxo25dihpyrazole_neut->an',
 '3oh6mercaptopyridazine_neut->an',
 '4oh26dimepyrimidine_cation->neut',
 '4oh26dimepyrimidine_neut->an',
 '29dime6mespurine_cation->neut',
 '67dime2mespurine_cation->neut',
 '69dime2mespurine_cation->neut',
 '5ipro46diohpyrimidine_cation->neut',
 '5ipro46diohpyrimidine_neut->an',
 '5me46diohpyrimidine_cation->neut',
 '5me46diohpyrimidine_neut->an',
 '5ipro4oh6mercpyrimidine_cation->neut',
 '5ipro4oh6mercpyrimidine_neut->an',
 '5ipro4oh6mercpyrimidine_an->2an',
 '5me6mercapto4ohpyrimidine_cation->neut',
 '5me6mercapto4ohpyrimidine_neut->an',
 '5me6mes4ohpyrimidine_cation->neut',
 '25dime6oh4ampyrimidine_cation->neut',
 '5me4ohpyrimidine_cation->neut',
 '5me4ohpyrimidine_neut->an',
 '6mercapto4ohpyrimidine_neut->an',
 '6mercapto4ohpyrimidine_an->2an',
 '6mes4ohpyrimidine_cation->neut',
 '6meo4ohpyrimidine_cation->neut',
 '2me46diohpyrimidine_cation->neut',
 '2me46diohpyrimidine_neut->an',
 '18dime2acm18dime16dih6oxopurine_cation->neut',
 '18dime2acm18dime16dih6oxopurine_neut->an',
 '6oh2acm78dimepurine_neut->an',
 '6oh2acm89dimepurine_neut->an',
 '6oh2acm8mepurine_cation->neut',
 '6oh2acm8mepurine_neut->an',
 '18dime2am18dime16dih6oxopurine_cation->neut',
 '18dime2am18dime16dih6oxopurine_neut->an',
 '18dime2am1me8cf316dih6oxopurine_cation->neut',
 '18dime2am1me8cf316dih6oxopurine_neut->an',
 '6oh2am78dimepurine_cation->neut',
 '6oh2am78dimepurine_neut->an',
 '6oh2am8mepurine_cation->neut',
 '6oh2am8mepurine_neut->an',
 '6oh2am8mepurine_an->2an',
 '6oh2am7me8cf3purine_cation->neut',
 '6oh2am7me8cf3purine_neut->an',
 '6oh2am8cf3purine_cation->neut',
 '6oh2am8cf3purine_neut->an',
 '6oh2am8cf3purine_an->2an',
 '6oh2am1mepurine_cation->neut',
 '6oh2am1mepurine_neut->an',
 '6oh2am7mepurine_cation->neut',
 '6oh2am7mepurine_neut->an',
 '6am3mepurine_cation->neut',
 '6am7mepurine_cation->neut',
 'pyrazolo34bpyrazine_cation->neut',
 '56dimepyrazolo34bpyrazine_cation->neut',
 '1mepyrazolo34bpyrazine_cation->neut',
 '156trimepyrazolo34bpyrazine_cation->neut',
 '6am123triazolo5p4p45pyrimidine_cation->neut',
 '6am123triazolo5p4p45pyrimidine_neut->an',
 '3pme6am123triazolo5p4p45pyrimidine_cation->neut',
 '6oh123triazolo5p4p45pyrimidine_neut->an',
 '6sh123triazolo5p4p45pyrimidine_neut->an',
 '6am2pme123triazolo5p4p45pyrimidine_cation->neut',
 '6am1pme123triazolo5p4p45pyrimidine_cation->neut',
 '2meampurine_cation->neut',
 '2meampurine_neut->an',
 '146triazaindene_cation->neut',
 '23dih5oimidazo12cpyrimidine_cation->neut',
 '23dih5oimidazo12cpyrimidine_neut->an',
 '34dihpteridine_cation->neut',
 '2me34dihpteridine_cation->neut',
 '6me34dihpteridine_cation->neut',
 '1234tethpteridine_2cation->cation',
 '1234tethpteridine_cation->neut',
 '4carbmquinazoline_cation->neut',
 '1ph5me23dih3oxopyrazole_cation->neut',
 '1ph5me23dih3oxopyrazole_neut->an',
 '6meo2oxo12dihpyridine_cation->neut',
 '6meo2oxo12dihpyridine_neut->an',
 '24dicarbxpyridine_neut->an',
 '24dicarbxpyridine_an->2an',
 '6cl2oxo12dihpyridine_cation->neut',
 '6cl2oxo12dihpyridine_neut->an',
 '2cl4ohpyridine_cation->neut',
 '2cl4ohpyridine_neut->an',
 '4phcarbamoylimidazole_cation->neut',
 '4phcarbamoylimidazole_neut->an',
 '25thiazolylpyridine_cation->neut',
 '5ph2pheam134oxadiazole_cation->neut',
 '4am5carbm123triazole_cation->neut',
 '4am5carbm123triazole_neut->an',
 '2me4am5carbm123triazole_cation->neut',
 '5678tetrahcinnoline_cation->neut',
 '34dime5678tetrahcinnoline_cation->neut',
 '3me5678tetrahcinnoline_cation->neut',
 '6formylpurine_cation->neut',
 '6formylpurine_neut->an',
 '235tricl4oxo24dihpyridine_neut->an',
 '4formylimidazole_cation->neut',
 '4formylimidazole_neut->an',
 '1me3oh6oxo16dihpyridazine_neut->an',
 '4carboxythiazolidine_cation->neut',
 '4carboxythiazolidine_neut->an',
 '55dime4carboxythiazolidine_cation->neut',
 '55dime4carboxythiazolidine_neut->an',
 '4me6oh23dihfuro23bpyridine_cation->neut',
 '4me6oh23dihfuro23bpyridine_neut->an',
 '2meam6ohpurine_cation->neut',
 '2meam6ohpurine_neut->an',
 '2meam6ohpurine_an->2an',
 '2h6h15methbenzob15diazocine_cation->net',
 '11me2h6h15methbenzob15diazocine_cation->net',
 '23dih5h14methanobenzoe14diazepine_cation->neut',
 '5br110phenanthroline_cation->neut',
 '1mepchlorophenylhydrazine_cation->neut',
 '1memchlorophenylhydrazine_cation->neut',
 '1membromophenylhydrazine_cation->neut',
 '1mepbromophenylhydrazine_cation->neut',
 '1mepmethoxyphenylhydrazine_cation->neut',
 '1meotolylhydrazine_cation->neut',
 '1memtolylhydrazine_cation->neut',
 '1meptolylhydrazine_cation->neut',
 '1mephenylhydrazine_cation->neut',
 '1me24dimephenylhydrazine_cation->neut',
 '1me3vbr4mephenylhydrazine_cation->neut',
 '2me123triazolo5p4p45pyrimidine_cation->neut',
 '5f2ambenzoic_cation->neut',
 '5f2ambenzoic_neut->an',
 '4me2ambenzoic_cation->neut',
 '4me2ambenzoic_neut->an',
 '5me2ambenzoic_cation->neut',
 '5me2ambenzoic_neut->an',
 '5meo2ambenzoic_cation->neut',
 '5meo2ambenzoic_neut->an',
 '4me6clpyridine21hone_cation->neut',
 '4me6clpyridine21hone_neut->an',
 '2am5ohbenzoic_cation->neut',
 '2am5ohbenzoic_neut->an',
 '3br12benzenedicarboxylic_neut->an',
 '3br12benzenedicarboxylic_an->2an',
 '3cl12benzenedicarboxylic_an->2an',
 '2mequinolinol_cation->neut',
 '2mequinolinol_neut->an',
 '5mercaptopyrimidine241h3hdione_neut->an',
 '5mercaptopyrimidine241h3hdione_an->2an',
 'pteridine261h5hdione_neut->an',
 'pteridine261h5hdione_an->2an',
 'pteridine461h5hdione_neut->an',
 'pteridine461h5hdione_an->2an',
 '1me2681h3h7htrionepurine_neut->an',
 '1me2681h3h7htrionepurine_an->2an',
 '7me2681h3h7htrionepurine_neut->an',
 '7me2681h3h7htrionepurine_an->2an',
 '8cf3purin21hone_neut->an',
 '8cf3purin21hone_an->2an',
 '19dime36dih6thioxopurin21hone_neut->an',
 '1me6mespurin21hone_cation->neut',
 '1me6mespurin21hone_neut->an',
 '7me6mespurin21hone_cation->neut',
 '9me6mespurin21hone_cation->neut',
 '9me6mespurin21hone_neut->an',
 '3me6mespurin23hone_cation->neut',
 '3me6mespurin23hone_neut->an',
 '1me2mespurin61hone_neut->an',
 '3nitro12benzenedicarboxylic_neut->an',
 '3nitro12benzenedicarboxylic_an->2an',
 '4meocarb2pyridinecarboxylic_neut->an',
 '5meocarb2pyridinecarboxylic_neut->an',
 '6meocarb2pyridinecarboxylic_neut->an',
 '1me28bismespurin61hone_cation->neut',
 '1me28bismespurin61hone_neut->an',
 '3me28bismespurin63hone_cation->neut',
 '1ph3ohpyrazole_neut->an',
 '23quinaxolinedicarboxylic_neut->an',
 '6me76him1h123triazolo45dpyrimidin_cation->neut',
 '8mercaptopurine_neut->an',
 '8mercaptopurine_an->2an',
 'thiopurinol_cation->neut',
 'thiopurinol_neut->an',
 '3ohpyridin21Hone_neut->an',
 '5ohpyridin21Hone_neut->an',
 '1me5am1h123triazolo45dpyrmdin74hone_cation->neut',
 '1me5am1h123triazolo45dpyrmdin74hone_neut->an',
 '2me5am1h123triazolo45dpyrmdin76hone_cation->neut',
 '2me5am1h123triazolo45dpyrmdin76hone_neut->an',
 '2clbenzoquinoneoxime_neut->an',
 '3clbenzoquinoneoxime_neut->an',
 '2brbenzoquinoneoxime_neut->an',
 '3brbenzoquinoneoxime_neut->an',
 '2ohbenzaldehydeoxime_neut->an',
 '2mebenzoquinoneoxime_neut->an',
 '3mebenzoquinoneoxime_neut->an',
 '45dimeisoxazol32hone_neut->an',
 '5cf3141h3hdionepyrimidine_neut->an',
 '5me24hexanedioneenol_neut->an',
 '268trimepurine_cation->neut',
 '268trimepurine_neut->an',
 '14dih4oxo26pyridinecarboxylic_neut->an',
 '3me7ampyrimidino54e124triazin51hone_cation->neut',
 '3me7ampyrimidino54e124triazin51hone_neut->an',
 '1me6thioxo36dihpurin21hone_neut->an',
 '1me6thioxo36dihpurin21hone_an->2an',
 '9me6thioxo36dihpurin21hone_neut->an',
 '1me8mercaptopurin21hone_neut->an',
 '9me8mercaptopurin21hone_neut->an',
 '9me8mercaptopurin21hone_an->2an',
 '6mespurin21hone_cation->neut',
 '6mespurin21hone_neut->an',
 '6mespurin21hone_an->2an',
 '3oh4oxohpyran26dicarbx_neut->an',
 '3oh4oxohpyran26dicarbx_an->2an',
 '8meam1imeamnaphthalene_cation->neut',
 '3am4nitropyrazole_cation->neut',
 '4mesulfonylpyridazine_cation->neut',
 '4cl6dimeam2ampyrimidine_cation->neut',
 '3me5nitro6oxo24diam36dihpyrimidine_cation->neut',
 '3me6oxo45diam36dihpyrimidine_cation->neut',
 '24diam6dimeampyrimidine_2cation->cation',
 '24diam6dimeampyrimidine_cation->neut',
 '6meam24diampyrimidine_2cation->cation',
 '6meam24diampyrimidine_cation->neut',
 '1me6mes4oxo14dihpyrimidine_cation->neut',
 '1me6thio4oh15dihpyrimidine_neut->an',
 '1me54thio6oh14dihpyrimidine_neut->an',
 '1me6thio4dimeam16dihpyrimidine_cation->neut',
 '5sulphonic8oh7nitroquinoline_neut->an',
 '5sulphonic8oh7nitroquinoline_an->2an',
 '5nitro4dimeam6meampyrimidine_cation->neut',
 '4me6oh5cn23dihfuro23bpyridine_neut->an',
 'trans10h16naphthyridine_2cation->cation',
 'trans10h16naphthyridine_cation->neut',
 'trans10h17naphthyridine_2cation->cation',
 'trans10h17naphthyridine_cation->neut',
 '5678tetrah17naphthyridine_2cation->neut',
 '5678tetrah17naphthyridine_cation->neut',
 '3nitro15naphthyridine_cation->neut',
 '24dione6789tethpyrdo12apyrmdin_cation->neut',
 '4oh34dih138triazanaphthalene_cation->neut',
 '6meam9mepurine_cation->neut',
 '9me2piperidinopurine_cation->neut',
 '9me6piperidinopurine_cation->neut',
 '9me8piperidinopurine_cation->neut',
 '7hpurin8ol_neut->an',
 '7hpurin8ol_an->2an',
 '26dioxo1236teth4pyrimidinecarboxylic_an->2an',
 '24dioxo1234teth5pyrimidinecarboxylic_an->2an',
 '5am26dio1236teth3pyrimidinecarboxylic_neut->an',
 'ibotenic_neut->an',
 '3ohisoxazole_neut->an',
 '3shisoxazole_neut->an',
 '2mes68dimepurine_2cation->cation',
 '2mes68dimepurine_cation->neut',
 '2mes68dimepurine_neut->an',
 '28bismes6mepurine_cation->neut',
 '28bismes6mepurine_neut->an',
 '2oh5678tetrhquinazolinecarboxylic_neut->an',
 '2oh5678tetrhquinazolinecarboxylic_an->2an',
 '2am5678tetrhquinazolinecarboxylic_cation->neut',
 '2am5678tetrhquinazolinecarboxylic_neut->an',
 '2me5678tetrhquinazolin41hone_cation->neut',
 '2me5678tetrhquinazolin41hone_neut->an',
 '13naphtalenediol_neut->an',
 '7i8oh5quinolinesulfonic_neut->an',
 '7i8oh5quinolinesulfonic_an->2an',
 '77dime3789tetrhpyrmdno21ipurine_cation->neut',
 '99dime3789tetrhpyrmdno21ipurine_cation->neut',
 '67dioh2naphthalenesulfonic_an->2an',
 '67dioh2naphthalenesulfonic_2an->3an',
 '2accyclopentanone_neut->an',
 '6me2ohcyclohexen1one_neut->an',
 '2mes8mepurine_cation->neut',
 '2mes8mepurine_neut->an',
 '17dimepurine2681h3h7htrione_neut->an',
 '17dimepurine2681h3h7htrione_an->2an',
 '233trime1pyrolidiniol_cation->neut',
 '22dime135cyclohexanetrione_neut->an',
 '2accyclohexanone_neut->an',
 '3me2ambenzoic_neut->an',
 '5678tetrahquinazolin41hone_cation->neut',
 '5678tetrahquinazolin41hone_neut->an',
 '357trime465h7hdiopyrzl34dpyrmdn_neut->an',
 '5678tetrahquinazoline41hthione_cation->neut',
 '5678tetrahquinazoline41hthione_neut->an',
 '89dime261h3hdionepurine_neut->an',
 '28bismespurin61hone_cation->neut',
 '28bismespurin61hone_neut->an',
 '28bismespurin61hone_an->2an',
 '2me5678tetrah41hthionequinazoline_cation->neut',
 '2me5678tetrah41hthionequinazoline_neut->an',
 '3cl5678tetraahisoquinolin12hone_cation->neut',
 '3cl5678tetraahisoquinolin12hone_neut->an',
 '2am345678hxh4quinazolinecarbx_cation->neut',
 '5sulpho7cl8ohquinoline_neut->an',
 '5sulpho7cl8ohquinoline_an->2an',
 '3oxo2356tetrah1himidazo12aimidazole_cation->neut',
 '1me1himidazo45bpyridine_cation->neut',
 '3o2367tetrh1h5himdazo12apyrimdne_cation->neut',
 '2367tetrh1h5himdazo12apyrimdne_cation->neut',
 '1me28dione1h7hpurine_neut->an',
 '34diohbenzenesulfonic_an->2an',
 '34diohbenzenesulfonic_2an->3an',
 '8oh24dimequinazollin_cation->neut',
 '4me8ohquinazoline_cation->neut',
 '3mesulfinylpyridazine_cation->neut',
 '4mesulfinylpyridazine_cation->neut',
 '45dime16diohpyridin21hone_neut->an',
 '4iodobenzimidazole_cation->neut',
 '5iodobenzimidazole_cation->neut',
 '2am4cl6meampyrimidine_cation->neut',
 '4am6cl2dimeampyrimidine_cation->neut',
 '5nitro2am4dimeam6ohpyrimidine_cation->neut',
 '5nitro2am4meam6ohpyrimidine_cation->neut',
 '6thioxo36dihpurin21hone_neut->an',
 '6thioxo36dihpurin21hone_an->2an',
 '6cn5am1h123triazolo45bpyridine_cation->neut',
 '12cyclohexanedione_neut->an',
 '34dinitro12benzenediol_neut->an',
 '34dinitro12benzenediol_an->2an',
 '3oh2pyridinecarbx_neut->an',
 '3oh2pyridinecarbx_an->2an',
 '125678hexah26dioxo4crbxpteridine_an->2an',
 '1ph2pyrazolin5one_neut->an',
 '2s5ph6oh23dihpyrimidin41hone_neut->an',
 '3ph7ampyrimidino54e124triazin5one_cation->neut',
 '3ph7ampyrimidino54e124triazin5one_neut->an',
 '435cicl4ohphenyl6me2cl25cyclohexadien1one_neut->an',
 '5pho2ambenzoic_cation->neut',
 '5pho2ambenzoic_neut->an',
 '435dibr4ohpheny3mel25cyclohexadien1oneimino_neut->an',
 '435dicl4ohph25dime25cyclohexadien1one_neut->an',
 '1ph123triazole45dicarboxylic_neut->an',
 '2clbenzohydroxamic_neut->an',
 '2fbenzohydroxamic_neut->an',
 'benzenetriol_neut->an',
 '7oxo47dih1hpyrazolo43dpyrimidine3crbx_neut->an',
 '111trifluro24pentanedione_neut->an',
 '3ph24pentanedione_neut->an',
 '13diphe13propanedione_neut->an',
 '1me4nitro5piperidinoimidazole_cation->neut',
 'cyanomelamine_cation->neut',
 '246triaminophenol_3cation->2cation',
 '246triaminophenol_2cation->cation',
 '246triaminophenol_cation->neut',
 '5ohimino246trione1h3h5hpyrimidine_neut->an',
 '4nitrosophenol_neut->an',
 '4nitroso26dimephenol_neut->an',
 '4nitroso35dimethylphenol_neut->an',
 '8oh7nitroso2naphthalenesulfonic_an->2an',
 '5oh6nitroso1naphthalenesulfonic_an->2an',
 '6oh5nitroso2naphthalenesulfonic_an->2an',
 'nndimethyl4nitroso_cation->neut',
 '5nitroso8ohquinoline_cation->neut',
 '5nitroso8ohquinoline_neut->an',
 '9amino2hydroxyacridine_cation->neut',
 '9amino4hydroxyacridine_cation->neut',
 '5ac6me2oxo1hpyridine3carbx_neut->an',
 '5ac6me2oxo1hpyridine3carbx_an->2an',
 '3meo6mespyridazine_cation->neut',
 '43cl4ohphenyl26dicl25cyclohexandien1oneimino_neut->an',
 '43br4ohphenyl26dibr25cyclohexandien1oneimino_neut->an',
 '435dicl4ohphenyl25cyclohexadien1oneimino_neut->an',
 '435dibr4ohphenyl25cyclohexadien1oneimino_neut->an',
 '435dicl4ohph3cl25cyclhexdien1oneimino_neut->an',
 '435dibr4ohph3br25cyclhexdien1oneimino_neut->an',
 '435dicl4ohph3me25cyclhexdien1oneimino_neut->an',
 '7me27diazaspiro35nonane_2cation->cation',
 '7me27diazaspiro35nonane_cation->neut',
 '7me17diazaspiro35nonane_2cation->cation',
 '7me17diazaspiro35nonane_cation->neut',
 '5methoxybenzimidazole_cation->neut',
 '5nitrobenzimidazole_cation->neut',
 '28bismes6mepurine_2cation->cation',
 '2mes8mepurine_2cation->cation',
 '23diampropanoic_cation->neut',
 '23diampropanoic_neut->an',
 '26diclpurine_cation->neut',
 '6oh145triazanaphthalene_cation->neut',
 '4me6oh5cn23dihfuro23bpyridine_cation->neut',
 '4mes6meopyrimidine_cation->neut',
 '5me6meo4mercaptpyrimidine_cation->neut',
 '6mercapto4ohpyrimidine_cation->neut',
 '1me54thio6oh14dihpyrimidine_cation->neut',
 '45diaminopyrimidine_2cation->cation',
 '1me3oh6oxo16dihpyridazine_cation->neut',
 '3me4nitropyrazole_cation->neut',
 '2oh34diampyridine_2cation->cation',
 '2oh5ampyridine_2cation->cation',
 '6sh7me8azapurine_cation->neut',
 '6am9me8azapurine_cation->neut',
 '6me8oxypterin_cation->neut',
 '6me8oxypterin_neut->an',
 '7me8oxypterin_cation->neut',
 '7me8oxypterin_neut->an',
 '67dime8oxypterin_cation->neut',
 '67dime58dioxypterin_cation->neut',
 '67dime58dioxypterin_neut->an',
 '7tertbutylpterin_cation->neut',
 '7tertbutylpterin_neut->an',
 '7tertbutyl5oxypterin_cation->neut',
 '7tertbutyl5oxypterin_neut->an',
 '6phpterin_cation->neut',
 '6phpterin_neut->an',
 '6phoxypterin_cation->neut',
 '7phpterin_cation->neut',
 '7phpterin_neut->an',
 '167trime5oxypterin_cation->neut',
 '247triamino6phpteridin_2cation->cation',
 '67diphepterin_cation->neut',
 '67diphepterin_neut->an',
 '67diphe8oxypterin_cation->neut',
 '67diphe8oxypterin_neut->an',
 '67diphe58dioxypterin_cation->neut',
 '67diphe58dioxypterin_neut->an',
 '5chlorouracil_neut->an',
 '5iodouracil_neut->an',
 '45diaminoacridine_2cation->cation',
 '68dihydroxypurine_neut->an',
 '7me68dioxypurin_neut->an',
 'benzoguanamine_cation->neut',
 '24diam1hpyrrolo23dpyrimidine_cation->neut',
 '24diam64clphsulfanylpyrimidine_cation->neut',
 '24diam5pbrphthieno23dpyrimidine_cation->neut',
 '24diam5me6phthieno23dpyrimidine_cation->neut',
 '4mercaptopyridazine_cation->neut',
 '2methylthioquinazoline_cation->neut',
 '2methiazole5carboxylic_neut->an',
 '5methiazole2carboxylic_neut->an',
 'thiazole2carboxylic_neut->an',
 'bromcyclizine_cation->neut',
 'fluorcyclizine_cation->neut',
 'methylcyclizine_cation->neut',
 '7me4am7hpyrrolo23dpyrimidine_cation->neut']



import sys
sys.path.insert(0,"../../import")
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
for c in columns_with_changes[0:80]:
    
    sct=go.Scatter(y=list(data_only1st[c]),x=list(data_standard[c]),mode='markers',showlegend=False,text=text,
                                marker=dict(color="red", line=dict(width=1),showscale=False) )
    #hist=go.Histogram(x=list( (data_standard[c]-data_only1st[c])/data_standard[c]),opacity=1.0,marker_color="red",
    #                  name=c)

    scatters.append(sct)
    #histograms.append(hist)
    


from plotly.subplots import make_subplots

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
"""                 
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



