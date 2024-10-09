import torch 
import numpy as np
import sklearn
import sklearn.manifold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def trim_collate(batch, audio_window_length=20480):
    audio_data = []
    speaker_ids = []
    for data in batch:
        waveform, _, _, speakerid, _, _ = data
        waveform = waveform.squeeze()
        part_index = np.random.randint(len(waveform) - audio_window_length + 1)
        audio_data.append(waveform[part_index:(part_index+audio_window_length)])
        speaker_ids.append(speakerid)
    return torch.stack(audio_data), torch.tensor(speaker_ids, dtype=torch.int)

############# VISUALIZATION UTILITY FUNCTIONS #############

def visualize_tsne(x, y_label, num_components=2, perplexity=5,
                   fig=None, ax=None, ax2=None, title_str=''):
  # perplexity = np.sqrt(n_points)) # TODO(loganesian): Evaluate default?
  tsne_tool = sklearn.manifold.TSNE(n_components=num_components, perplexity=perplexity)
  pca_tool = sklearn.decomposition.PCA(n_components=num_components)
  z_this = tsne_tool.fit_transform(x)
  pca_z_this = pca_tool.fit_transform(x)

  df = pd.DataFrame({'label' : y_label.flatten()})
  df = df.dropna(subset=['label'])
  df['row_id'] = range(0, len(df))
  df['z_1'], df['z_1_pca'] = z_this[:, 0], pca_z_this[:, 0]
  if num_components == 2:
    df['z_2'], df['z_2_pca'] = z_this[:, 1], pca_z_this[:, 1]
  n_cat = len(pd.unique(df['label']))

  if fig is None or ax is None:
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(121)
  if num_components == 2:
    scatter_ax = sns.scatterplot(x='z_1', y='z_2', data=df, hue='label', s=30,
                                 palette=sns.color_palette("Set2", n_cat))
  else:
    scatter_ax = sns.scatterplot(x='z_1', y='z_1', data=df, hue='label', s=30,
                                 palette=sns.color_palette("Set2", n_cat))
  scatter_ax.set(title=f'{title_str} t-SNE Perplexity: {perplexity}, Components: {num_components}')

  if ax2 is None:
      ax2 = fig.add_subplot(122)
  if num_components == 2:
    scatter_ax = sns.scatterplot(x='z_1_pca', y='z_2_pca', data=df, hue='label',
                                 s=30, palette=sns.color_palette("Set2", n_cat))
  else:
    scatter_ax = sns.scatterplot(x='z_1_pca', y='z_1_pca', data=df, hue='label',
                                 s=30, palette=sns.color_palette("Set2", n_cat))
  scatter_ax.set(title=f'{title_str} PCA Components: {num_components}')
  return fig, ax, ax2