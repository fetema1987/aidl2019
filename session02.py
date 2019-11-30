
import tensorflow as tf
import matplotlib as plt
#import session01
from session01 import DataDistribution
def main(lr=None):
    # TODO: E02: Define Linear Regressor graph --> Definition phase (use graph, placeholder, variable, operations)
    graph = tf.Graph()
    with graph.as_default():

        w = tf.get_variable('w', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        b = tf.get_variable('b', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        x = tf.placeholder(name='x', dtype=tf.float32)
        # Es dentro del gráfico donde tendremos el target y calcularemos la pérdida, gradientes y backprop
        y = tf.placeholder(name='x', dtype=tf.float32)
        lrate = lr
        # Calculamos el valor predecido
        z = w * x
        y_ = z + b

        # Error
        diff = y_ - y
        # Error cuadrático L = (y_ - y) ^2
        loss = tf.pow(diff, 2)
        # Derivada del error cuadrático respecto a la predicción dL = 2(y_ - y)
        loss_grad = 2 * diff

        # Usamos, a través del chain rule, el gradiente del error para backprop y actualizar el gradiente de los pesos
        w_grad = x * loss_grad # Operación de cálculo de nuevo gradiente, como es multiplicación, derivada es x por nuevo grad
        b_grad = 1 * loss_grad

        # Actualizamos los pesos mediante el LR y el nuevo gradiente del peso
        # Restamos (por descender en gráfico del error) al peso LR * nuevo gradiente
        w_update = w.assign(w - lrate * w_grad)
        b_update = b.assign(b - lrate * w_grad)

        # Se hace esto para agrupar oficialmente la salida de actualización del entrenamiento
        train_op = tf.group(w_update, b_update)



    # TODO: E03: Run a forward pass --> Run phase (use session and the DataDistribution class from previous exercise)
    with tf.Session(graph=graph) as sess:

        """
        i, j = data.test(1)
        result = sess.run(y_, feed_dict={x: i})
        print('Input : {} --> Prediction {}. Label{}'.format(i, result, j))
        """
        loss_hist = []
        data = DataDistribution()
        sess.run(tf.global_variables_initializer())
        for input_data, label in data(100):
            prediction, loss_val, _ = sess.run([y_, loss, train_op], feed_dict={x: input_data, y: label})
            # TODO: E04: Implement optimization step manually!
            print('Loss {}, Input : {} --> Prediction {}. Label {}, Loss {}'.format(loss_val, input_data, prediction, label))
        w_pred, b_pred = sess.run([W, b])
    # Sentencia para escribir la ejecución del gráfico y después leerla con Tensorboard
    writer = tf.summary.FileWriter(os.path.expanduser(logdir), graph=graph)


if __name__ == '__main__':
    main(lr=0.1)
