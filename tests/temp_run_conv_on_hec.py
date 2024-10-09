from tests.test_converter import test_converter_both

# learning_steps=250*2*3 is 53s on M1 MBP
test_converter_both(learning_steps=250*2*3*15)
