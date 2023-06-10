import matplotlib.pyplot as plt

def visualization(history):
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   epochs = range(1, len(loss) + 1)
   plt.plot(epochs, loss, 'y', label='Training loss')
   plt.plot(epochs, val_loss, 'r', label='Validation loss')
   plt.title('Training and validation loss')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.ylim(0, 1)  # Set the y-axis limits from 0 to 1
   plt.legend()
   plt.show()

   acc = history.history['acc']
   #acc = history.history['accuracy']
   val_acc = history.history['val_acc']
   #val_acc = history.history['val_accuracy']

   plt.plot(epochs, acc, 'y', label='Training acc')
   plt.plot(epochs, val_acc, 'r', label='Validation acc')
   plt.title('Training and validation accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.ylim(0, 1)  # Set the y-axis limits from 0 to 1
   plt.legend()
   plt.show()

   iou = history.history['iou']
   val_iou = history.history['val_iou']
   epochs = range(1, len(iou) + 1)
   plt.plot(epochs, iou, 'y', label='Training iou')
   plt.plot(epochs, val_iou, 'r', label='Validation iou')
   plt.title('Training and validation iou')
   plt.xlabel('Epochs')
   plt.ylabel('Iou')
   plt.ylim(0, 0.6)  # Set the y-axis limits from 0 to 1
   plt.legend()
   plt.show()
    
    
    