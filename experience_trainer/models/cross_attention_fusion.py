import torch
import torch.nn as nn


class CrossAttentionRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(CrossAttentionRegression, self).__init__()

        # Capas para el mecanismo de atención cruzada
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_value_projection = nn.Linear(input_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Cabeza de regresión
        self.regression_head = nn.Linear(hidden_dim,
                                         output_dim)  # output_dim es el número de valores continuos a predecir

    def forward(self, input1, input2):
        # Proyectar las entradas a la dimensión oculta
        query = self.query_projection(input1)  # Forma: (seq_len, batch, hidden_dim)
        key_value = self.key_value_projection(input2)  # Forma: (seq_len, batch, hidden_dim)

        # Aplicar atención cruzada
        attn_output, _ = self.cross_attention(query, key_value, key_value)

        # Agregar las salidas de atención; aquí, simplemente tomamos la media de la secuencia
        aggregated_output = attn_output.mean(dim=0)  # Forma: (batch, hidden_dim)

        # Capa de regresión
        output = self.regression_head(aggregated_output)  # Forma: (batch, output_dim)

        return output
