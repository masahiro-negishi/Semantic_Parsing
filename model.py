import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim

class Encoder(nn.Module):
    """Encoder

    Encoder without Attention

    """
    def __init__(self, emb_dim:int, hid_dim:int, v_size:int, device:int, num_layers:int,  batch_first:bool, dropout_rate:float):
        """init

        initialize encoder

        Args:
            emb_dim (int): the dimension of embedded word vector
            hid_dim (int): the dimension of hidden state of LSTM
            v_size (int): the size of the input vocabulary
            device (torch.device): cpu or cuda
            num_layers (int): the number of LSTM layers
            batch_first (bool): set to False
            dropout_rate (float): dropout rate between different LSTM layers 
        """
        super(Encoder, self).__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.embed = nn.Embedding(v_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=batch_first, dropout=dropout_rate, bidirectional=False) # many to one

    def forward(self, text:Tensor):
        """forward

        forward process

        Args:
            text (Tensor): a batch of input sentences
        
        Returns:
            state (Tensor): the last hidden state and cell state of LSTM
        """
        embedding = self.embed(text)  # (text_len, batch_size) -> (text_len, batch_size, emb_dim)
        _, state = self.lstm(embedding) # (text_len, batch_size, emb_dim) -> _,  ((num_layers, batch_size, hid_dim), (num_layers, batch_size, hid_dim)) # state = (h_n, c_n)
        return state


class Encoder_Attention(nn.Module):
    """Encoder

    Encoder with Attention

    """
    def __init__(self, emb_dim:int, hid_dim:int, v_size:int, device:int, num_layers:int,  batch_first:bool, dropout_rate:float):
        """init

        initialize encoder

        Args:
            emb_dim (int): the dimension of embedded word vector
            hid_dim (int): the dimension of hidden state of LSTM
            v_size (int): the size of the input vocabulary
            device (torch.device): cpu or cuda
            num_layers (int): the number of LSTM layers
            batch_first (bool): set to False
            dropout_rate (float): dropout rate between different LSTM layers 
        """
        super(Encoder_Attention, self).__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.embed = nn.Embedding(v_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=batch_first, dropout=dropout_rate, bidirectional=False) # many to one

    def forward(self, text:Tensor):
        """forward

        forward process

        Args:
            text (Tensor): a batch of input sentences
        
        Returns:
            output (Tensor): all the hidden states
            state (Tensor): the last hidden state and cell state of LSTM
        """
        embedding = self.embed(text)  # (text_len, batch_size) -> (text_len, batch_size, emb_dim)
        output, state = self.lstm(embedding) # (text_len, batch_size, emb_dim) -> (text_len, batch_size, hid_dim),  ((num_layers, batch_size, hid_dim), (num_layers, batch_size, hid_dim)) # state = (h_n, c_n)
        return output, state


class Decoder(nn.Module):
    """Decoder

    Decoder without Attention
    
    """
    def __init__(self, emb_dim:int, hid_dim:int, v_size:int, device:int, num_layers:int,  batch_first:bool, dropout_rate:float, output_dropout_rate:float):
        """init

        initialize decoder

        Args:
            emb_dim (int): the dimension of embedded word vector
            hid_dim (int): the dimension of hidden state of LSTM
            v_size (int): the size of the output vocabulary
            device (torch.device): cpu or cuda
            num_layers (int): the number of LSTM layers
            batch_first (bool): set to False
            dropout_rate (float): dropout rate between different LSTM layers 
            output_dropout_rate (float): dropout rate after the last LSTM layer
        """
        super(Decoder, self).__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.v_size = v_size
        self.embed = nn.Embedding(v_size, hid_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=batch_first, dropout=dropout_rate, bidirectional=False) # many to many
        self.dropout = nn.Dropout(output_dropout_rate)
        self.linear = nn.Linear(hid_dim, v_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input:Tensor, hidden:Tensor):
        """forward

        forward one step

        Args:
            input (Tensor): input word at a time
            hidden (Tensor): the last hidden state and cell state of the encoder

        Returns:
            output (Tensor): output distribution
            hidden (Tensor): hidden state and cell state 
        """
        embedding = self.embed(input) # (batch_size) -> (batch_size, hid_dim)
        output, hidden = self.lstm(embedding.unsqueeze(0), hidden) # ((1, batch_size, hid_dim), (h, c)) -> ((1, batch_size, hid_size), (h, c))
        output = self.dropout(output.squeeze(0)) # (1, batch_size, hid_size) -> (batch_size, hid_size)
        output = self.linear(output) # (batch_size, hid_size) -> (batch_size, v_size)
        output = self.softmax(output) # (batch_size, v_size) -> (batch_size, v_size)
        return output, hidden


class Decoder_Attention(nn.Module):
    """Decoder

    Decoder with Attention
    
    """
    def __init__(self, emb_dim:int, hid_dim:int, v_size:int, device:int, num_layers:int,  batch_first:bool, dropout_rate:float, output_dropout_rate:float):
        """init

        initialize decoder

        Args:
            emb_dim (int): the dimension of embedded word vector
            hid_dim (int): the dimension of hidden state of LSTM
            v_size (int): the size of the output vocabulary
            device (torch.device): cpu or cuda
            num_layers (int): the number of LSTM layers
            batch_first (bool): set to False
            dropout_rate (float): dropout rate between different LSTM layers 
            output_dropout_rate (float): dropout rate after the last LSTM layer
        """
        super(Decoder_Attention, self).__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.v_size = v_size
        self.embed = nn.Embedding(v_size, hid_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=batch_first, dropout=dropout_rate, bidirectional=False) # many to many
        self.W1 = torch.nn.Linear(hid_dim, hid_dim, False)
        self.W2 = torch.nn.Linear(hid_dim, hid_dim, False)
        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(output_dropout_rate)
        self.linear = nn.Linear(hid_dim, v_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input:Tensor, hidden:Tensor, encoder_output:Tensor):
        """forward

        forward one step

        Args:
            input (Tensor): input word at a time
            hidden (Tensor): the last hidden state and cell state of the encoder
            encoder_output (Tensor): all the hidden states of the encoder

        Returns:
            output (Tensor): output distribution
            hidden (Tensor): hidden state and cell state 
        """
        embedding = self.embed(input) # (batch_size) -> (batch_size, hid_dim)
        output, hidden = self.lstm(embedding.unsqueeze(0), hidden) # ((1, batch_size, hid_dim), (h, c)) -> ((1, batch_size, hid_size), (h, c))
        text_len = encoder_output.shape[0]
        extend_output = output.repeat(text_len, 1, 1) # (text_len, batch_size, hid_dim)
        product = torch.bmm(extend_output.view(-1, 1, self.hid_dim), encoder_output.view(-1, self.hid_dim, 1)) # (text_len * batch_size, 1, 1)
        weight = product.view(text_len, -1, 1).repeat(1, 1, self.hid_dim) # (text_len, batch_size, hid_dim)
        weighted_encoder_output = weight * encoder_output # (text_len, batch_size, hid_dim)
        c = weighted_encoder_output.sum(dim=0) # (batch_size, hid_dim)

        output = self.Tanh(self.W1(output) + self.W2(c)) # (batch_size, hid_dim)

        output = self.dropout(output.squeeze(0)) # (1, batch_size, hid_size) -> (batch_size, hid_size)
        output = self.linear(output) # (batch_size, hid_size) -> (batch_size, v_size)
        output = self.softmax(output) # (batch_size, v_size) -> (batch_size, v_size)
        return output, hidden


class Seq2Seq(nn.Module):
    """Seq2seq model

    seq2seq model
    
    """
    def __init__(self, encoder: nn.Module, decoder:nn.Module, device:int, form_field, scheduled_sampling=1.0):
        """init

        Initialize seq2seq model

        Args:
            encoder (nn.Module): encoder
            decoder (nn.Module): decoder
            device (torch.device): cpu or cuda
            form_field (torchtext.data.Field): form_field
            scheduled_sampling (float): To what rate use golden answer while training
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.form_field = form_field
        self.scheduled_sampling = scheduled_sampling

    def train_forward(self, text:Tensor, form:Tensor) -> Tensor:
        """train forward

        train forward process

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
        
        Returns:
            outputs (Tensor): distributions of words at each time
        """
        batch_size = text.shape[1]
        max_len = form.shape[0]
        form_v_size = self.decoder.v_size
        
        outputs = torch.zeros(max_len, batch_size, form_v_size).to(self.device) # softmax outputs

        hidden = self.encoder(text) # (text_len, batch_size, emb_dim) -> ((num_layers, batch_size, hid_dim), (num_layers, batch_size, hid_dim))

        outputs[0] = torch.nn.functional.one_hot(form[0, :], num_classes=form_v_size)

        for t in range(1, max_len):
            ans = torch.where((torch.rand(batch_size) < self.scheduled_sampling).to(self.device), form[t-1, :], outputs[t-1].argmax(1))
            out, hidden = self.decoder(ans, hidden)
            outputs[t] = out 
        
        return outputs # (max_len, batch_size, form_v_size)

    def test_forward(self, text:Tensor, form:Tensor) -> Tensor:
        """test forward

        test forward process without beam search

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
        
        Returns:
            outputs (Tensor): distributions of words at each time
            predicts (Tensor): prediction
        """
        batch_size = text.shape[1]
        max_len = form.shape[0]
        form_v_size = self.decoder.v_size
        
        outputs = torch.zeros(max_len, batch_size, form_v_size) # softmax outputs
        predicts = torch.zeros(max_len, batch_size, dtype=int)

        hidden = self.encoder(text)

        outputs[0] = torch.nn.functional.one_hot(form[0, :], num_classes=form_v_size)
        predicts[0] = form[0, :]
        dec_in = form[0, :]

        for t in range(1, max_len):
            out, hidden = self.decoder(dec_in, hidden)
            outputs[t] = out
            dec_in = out.argmax(1)
            predicts[t] = dec_in
        
        return outputs, predicts

    def test_forward_with_ans(self, text:Tensor, form:Tensor) -> Tensor:
        """test forward

        test forward process with right answers without beam search
        (used for checking whether training is done properly)

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
        
        Returns:
            outputs (Tensor): distributions of words at each time
            predicts (Tensor): prediction
        """
        batch_size = text.shape[1]
        max_len = form.shape[0]
        form_v_size = self.decoder.v_size
        
        outputs = torch.zeros(max_len, batch_size, form_v_size) # softmax outputs
        predicts = torch.zeros(max_len, batch_size, dtype=int)

        hidden = self.encoder(text)

        outputs[0] = torch.nn.functional.one_hot(form[0, :], num_classes=form_v_size)
        predicts[0] = form[0, :]
        dec_in = form[0, :]

        for t in range(1, max_len):
            # out, hidden = self.decoder(dec_in, hidden)
            out, hidden = self.decoder(form[t-1, :], hidden)
            outputs[t] = out
            dec_in = out.argmax(1)
            predicts[t] = dec_in
        
        return outputs, predicts

    def test_forward_beam(self, text:Tensor, form:Tensor, keep_dim:int) -> Tensor:
        """test forward

        test forward process with beam search (batch size should be set to 1)

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
            keep_dim (int): how many candidates to keep at each time
        
        Returns:
            outputs (Tensor): distributions of words at each time
            predicts (Tensor): prediction
        """
        batch_size = text.shape[1]
        if batch_size != 1:
            raise ValueError(f'batch_size should be 1 but given batch_size is {batch_size}')
        form_v_size = self.decoder.v_size

        states = self.encoder(text) # ((num_layers, batch_size, hid_dim), (num_layers, batch_size, hid_dim))

        topk = [{
            'output': [self.form_field.vocab.stoi['<S>']],
            'log_prob': 0.0,
            'states': states
        }]

        for t in range(1, 50):
            candidate = []
            for keep in topk:
                # reach end of sentence
                if keep['output'][-1] == self.form_field.vocab.stoi['<E>']:
                    candidate.append(keep)
                    continue
                out, hidden = self.decoder(torch.tensor([keep['output'][-1]]).to(self.device), keep['states'])
                log_out = torch.log(out).squeeze(dim=0)
                for i in range(form_v_size):
                    candidate.append({
                        'output': keep['output'] + [i],
                        'log_prob': keep['log_prob'] + log_out[i].item(),
                        'states': hidden
                    })
            candidate.sort(key=lambda x: x['log_prob'], reverse=True)
            topk = candidate[0:keep_dim]
        
        return topk


class Seq2Seq_Attention(nn.Module):
    """Seq2seq model with attention

    seq2seq model with attention
    
    """
    def __init__(self, encoder: nn.Module, decoder:nn.Module, device:int, form_field, scheduled_sampling):
        """init

        Initialize seq2seq model

        Args:
            encoder (nn.Module): encoder
            decoder (nn.Module): decoder
            device (torch.device): cpu or cuda
            form_field (torchtext.data.Field): form_field
            scheduled_sampling (float): To what rate use golden answer while training
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.form_field = form_field
        self.scheduled_sampling = scheduled_sampling

    def train_forward(self, text:Tensor, form:Tensor) -> Tensor:
        """train forward

        train forward process

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
        
        Returns:
            outputs (Tensor): distributions of words at each time
        """
        batch_size = text.shape[1]
        max_len = form.shape[0]
        form_v_size = self.decoder.v_size
        
        outputs = torch.zeros(max_len, batch_size, form_v_size).to(self.device) # softmax outputs

        enc_output, hidden = self.encoder(text) # (text_len, batch_size, emb_dim) -> _, ((num_layers, batch_size, hid_dim), (num_layers, batch_size, hid_dim))

        outputs[0] = torch.nn.functional.one_hot(form[0, :], num_classes=form_v_size)

        for t in range(1, max_len):
            ans = torch.where((torch.rand(batch_size) < self.scheduled_sampling).to(self.device), form[t-1, :], outputs[t-1].argmax(1))
            out, hidden = self.decoder(ans, hidden, enc_output)
            outputs[t] = out 
        
        return outputs # (max_len, batch_size, form_v_size)

    def test_forward(self, text:Tensor, form:Tensor) -> Tensor:
        """test forward

        test forward process without beam search

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
        
        Returns:
            outputs (Tensor): distributions of words at each time
            predicts (Tensor): prediction
        """
        batch_size = text.shape[1]
        max_len = form.shape[0]
        form_v_size = self.decoder.v_size
        
        outputs = torch.zeros(max_len, batch_size, form_v_size) # softmax outputs
        predicts = torch.zeros(max_len, batch_size, dtype=int)

        enc_output, hidden = self.encoder(text)

        outputs[0] = torch.nn.functional.one_hot(form[0, :], num_classes=form_v_size)
        predicts[0] = form[0, :]
        dec_in = form[0, :]

        for t in range(1, max_len):
            out, hidden = self.decoder(dec_in, hidden, enc_output)
            # out, hidden = self.decoder(form[t, :], hidden)
            outputs[t] = out
            dec_in = out.argmax(1)
            predicts[t] = dec_in
        
        return outputs, predicts

    def test_forward_with_ans(self, text:Tensor, form:Tensor) -> Tensor:
        """test forward

        test forward process with right answers without beam search
        (used for checking whether training is done properly)

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
        
        Returns:
            outputs (Tensor): distributions of words at each time
            predicts (Tensor): prediction
        """
        batch_size = text.shape[1]
        max_len = form.shape[0]
        form_v_size = self.decoder.v_size
        
        outputs = torch.zeros(max_len, batch_size, form_v_size) # softmax outputs
        predicts = torch.zeros(max_len, batch_size, dtype=int)

        enc_output, hidden = self.encoder(text)

        outputs[0] = torch.nn.functional.one_hot(form[0, :], num_classes=form_v_size)
        predicts[0] = form[0, :]
        dec_in = form[0, :]

        for t in range(1, max_len):
            # out, hidden = self.decoder(dec_in, hidden)
            out, hidden = self.decoder(form[t-1, :], hidden, enc_output)
            outputs[t] = out
            dec_in = out.argmax(1)
            predicts[t] = dec_in
        
        return outputs, predicts

    def test_forward_beam(self, text:Tensor, form:Tensor, keep_dim:int) -> Tensor:
        """test forward

        test forward process with beam search (batch size should be set to 1)

        Args:
            text (Tensor): a batch of input sentences
            form (Tensor): a batch of output sentences
            keep_dim (int): how many candidates to keep at each time
        
        Returns:
            outputs (Tensor): distributions of words at each time
            predicts (Tensor): prediction
        """
        batch_size = text.shape[1]
        if batch_size != 1:
            raise ValueError(f'batch_size should be 1 but given batch_size is {batch_size}')
        max_len = form.shape[0]
        form_v_size = self.decoder.v_size

        enc_output, states = self.encoder(text) # ((num_layers, batch_size, hid_dim), (num_layers, batch_size, hid_dim))

        topk = [{
            'output': [self.form_field.vocab.stoi['<S>']],
            'log_prob': 0.0,
            'states': states
        }]

        for t in range(1, 50):
            candidate = []
            for keep in topk:
                # reach end of sentence
                if keep['output'][-1] == self.form_field.vocab.stoi['<E>']:
                    candidate.append(keep)
                    continue
                out, hidden = self.decoder(torch.tensor([keep['output'][-1]]).to(self.device), keep['states'], enc_output)
                log_out = torch.log(out).squeeze(dim=0)
                for i in range(form_v_size):
                    candidate.append({
                        'output': keep['output'] + [i],
                        'log_prob': keep['log_prob'] + log_out[i].item(),
                        'states': hidden
                    })
            candidate.sort(key=lambda x: x['log_prob'], reverse=True)
            topk = candidate[0:keep_dim]
        
        return topk
        

def init_weights(m: nn.Module):
    """init weights of a model

    init weights of a model

    Args:
        m (nn.Module): model
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.08, 0.08)
        else:
            nn.init.constant_(param.data, 0)


def generate_criterion_and_optimzier(ignore_index, parameters, lr, alpha):
    """generate criterion and optimizer

    generate criterion and optimzier

    Args:
        ignore_index (int): index to ignore when calculating cross entropy loss
        lr (float): learning rate
        alpha (float): hyperparameter for RMSprop
    
    Returns:
        criterion (nn.CrossEntropyLoss): cross entropy loss
        optimzier (nn.RMSprop): RMSprop
    """
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.RMSprop(parameters, lr=lr, alpha=alpha)
    return criterion, optimizer


def save_model(model, path):
    """save model

    save model

    Args:
        model (nn.Module): model to save
        path (string): where to save
    """
    torch.save(model.state_dict(), path)
