import argparse

parser = argparse.ArgumentParser(prog='demo', description='A demo program', epilog='The end of usage')
parser.print_help()
print('\n\n')



parser.add_argument('name')
parser.add_argument('-a', '--age', type=int, required=True)
parser.add_argument('-s', '--status', choices=['alpha', 'beta', 'released'], type=str, dest='myStatus')
parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
parser.print_help()
print('\n\n')



args = parser.parse_args()
print(args)
print()



args, unparsed = parser.parse_known_args()
print('args=%s, unparsed=%s' % (args, unparsed))
