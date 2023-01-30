import plotly.graph_objects as go
import pickle

dataset = 'avenue'

for i in range(21):
    with open(f'./logs/anomaly_score_{i+1}.txt', 'rb') as f:
        score = pickle.load(f)
        
        frame = []
        for j in range(len(score)):
            frame.append(j+1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame,
                                 y=1-score,
                                 name=f'Video {i+1}',
                                 line=dict(color='firebrick', width=2)
                                 )
                      )
        fig.update_layout(title=f'Anomaly score of video {i+1} in {dataset}',
                          xaxis_title='Frame',
                          yaxis_title='Anomaly score')
        fig.show()
        print()