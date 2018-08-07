import pickle
from ntn_plot import plot_multi

with open('results_train_val.pkl', 'rb') as fhand:
  ret = pickle.load(fhand)
  results = ret['accuracy_validation']
  results.sort(key=lambda x: x[0], reverse=True)
  print('validation accuracy:', results)
  print()

  train_losses, train_accs, val_losses, val_accs = ret['cost_training'], ret['accuracy_training'], ret['cost_validation'], ret['accuracy_validation']
  plot_multi([train_losses, val_losses], ['train', 'val'], 'loss', 'epoch', [train_accs, val_accs], ['train', 'val'], 'accuracy', 'epoch')

with open('results_test.txt', 'r') as fhand:
  results = [fhand.read()]
  results.sort(key=lambda x: x[0], reverse=True)
  print('testing accuracy:', results)
