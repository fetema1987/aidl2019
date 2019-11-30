
import tensorflow as tf
import matplotlib.pyplot as plt
#import session01
from session01 import DataDistribution
def main(lr=None):

    # Definimos el gráfico
    graph = tf.Graph()
    with graph.as_default():

        # Definimos placeholders
        x = tf.placeholder(name='x', shape=[], dtype=tf.float32)
        y = tf.placeholder(name='y', shape=[], dtype=tf.float32)

        # Definimos variables
        w = tf.get_variable(name='w', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        b = tf.get_variable(name='b', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())

        # Definimos constantes
        #lr = 0.001

        # FORWARD PASS
        z = x + w
        y_ = b + z
        # Hasta aqui la ejecución sin aprendizaje

        # A partir de aqui, calculamos el error, gradientes y backprop
        diff = y_ - y
        loss = tf.pow(diff, 2)

        # BACKWARD PASS
        # Calculamos el gradiente del Error respecto a los pesos
        loss_grad = 2 * diff

        # Actualizamos los gradientes de nuestras variables
        w_grad = x * loss_grad
        b_grad = 1 * loss_grad


        # OPTIMIZATION
        # Actualizamos nuestras variables para aprender. Si lo hacemos sin assign, NO estaremos actualizandolo de verdad.
        #w_update = w.assign(w - lr * w_grad)
        #b_update = b.assign(b - lr * b_grad)
        #train_op = tf.group(w_update, b_update)

        # La forma normal de hacerlo (no manualmente), es la siguiente. Definimos una op de SGD, y supongo que por estar dentro
        # del gráfico ya sabe leer nuestras variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss)


    with tf.Session(graph=graph) as sess:
        loss_hist=[]
        data = DataDistribution()
        sess.run(tf.global_variables_initializer())
        # Creamos unos datos e iteramos en el generator
        for input_data, label in data(5000):
            prediction, error, _ = sess.run([y_, loss, train_op], feed_dict={x:input_data, y:label})
            print('Prediccion {}, Target {}, Error {}'.format(prediction, label, error))
            loss_hist.append(error)
        W_pred, b_pred = sess.run([w, b])
        print('W GT: {}. W pred: {}'.format(data.w, W_pred))
        print('b GT: {}. b pred: {}'.format(data.b, b_pred))
        plt.scatter(loss_hist, range(0, 5000))
    #writer = tf.summary.FileWriter(os.path.expanduser(logdir), graph=graph)
    pass


if __name__ == '__main__':
    main(lr=0.0001)
