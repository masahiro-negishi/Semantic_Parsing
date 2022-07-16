import torchtext
import csv
import pandas as pd


def sentence_level_accuracy(ans, predict):
    """sentence level accuracy

    check whether the ans and predict match perfectly

    Args:
        ans (list of int): answer formula
        predict (list of int): predicted formula

    Returns:
        int: if perfect match then 1 else 0
    """
    if len(ans) != len(predict):
        return 0
    for i in range(len(ans)):
        if ans[i] != predict[i]:
            return 0
    return 1


def evaluate(seq2seq, iterator_beam, device, form_field, dataset_size):
    """evaluate model

    evaluate model

    Args:
        seq2seq (nn.Module): model
        iterator_beam (torchtext.data.Iterator): iterator for beam search
        device (torch.device): cpu or cuda
        form_field (torchtext.data.Field): Field for output
        dataset_size (int): dataset size
    """
    seq2seq.eval()
    total = 0
    sentence_match = 0
    candidate_corpus = []
    reference_corpus = []
    for batch in iterator_beam:
        text = batch.Text[0].to(device)
        form = batch.Form[0].to(device)

        topk = seq2seq.test_forward_beam(text, form, 10)
        
        candidate = list(form.to('cpu').squeeze())
        reference = topk[0]['output']
        candidate_corpus.append(list(map(lambda x: form_field.vocab.itos[x], candidate)))
        reference_corpus.append([list(map(lambda x: form_field.vocab.itos[x], reference))])
        
        sentence_match += sentence_level_accuracy(candidate, reference)
        total += 1
        if total % 10 == 0:
            print(f"{total} / {dataset_size}")
    print("Sentence level accuracy: ", sentence_match / total)
    print("Bleu score:", torchtext.data.metrics.bleu_score(candidate_corpus, reference_corpus))


def evaluate_and_output_to_csv(seq2seq, iterator_beam, device, form_field, dataset_size, output_file, output_column_name):
    seq2seq.eval()
    total = 0
    sentence_match = 0
    candidate_corpus = []
    reference_corpus = []

    results = []
    forms = []
    for batch in iterator_beam:
        text = batch.Text[0].to(device)
        form = batch.Form[0].to(device)

        topk = seq2seq.test_forward_beam(text, form, 10)
        
        candidate = list(form.to('cpu').squeeze())
        reference = topk[0]['output']
        candidate_corpus.append(list(map(lambda x: form_field.vocab.itos[x], candidate)))
        reference_corpus.append([list(map(lambda x: form_field.vocab.itos[x], reference))])
        
        match = sentence_level_accuracy(candidate, reference)
        sentence_match += match
        if match == 0:
            results.append('x')
        else:
            results.append('o')
        forms.append(" ".join(list(map(lambda x: form_field.vocab.itos[x], reference))[1:-1]))
        total += 1
        if total % 10 == 0:
            print(f"{total} / {dataset_size}")
    print("Sentence level accuracy: ", sentence_match / total)
    print("Bleu score:", torchtext.data.metrics.bleu_score(candidate_corpus, reference_corpus))

    df = pd.read_csv(filepath_or_buffer=output_file, sep=',', header=0)
    df[f'{output_column_name}_result'] = results
    df[f'{output_column_name}_forms'] = forms
    df.to_csv(output_file, ',')
