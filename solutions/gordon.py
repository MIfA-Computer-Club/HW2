#! /usr/bin/env python
import pyfits
import numpy as np
import argparse

def profile(fname,ap,skyap):
    pass

def main():
    parser = argparse.ArgumentParser(description='Plot radial profiles of input images.')

    parser.add_argument('files',nargs='+',help='Input files')
    parser.add_argument('-ap',nargs='+',required=True,type=float,help='Size [pixels] of object aperture')
    parser.add_argument('-skyap',nargs='+',required=True,type=float,help='Size [pixels] of sky aperture')

    args = parser.parse_args()

    if len(args.ap) is not in [1,len(args.files)]:
        raise ValueError('Length of aperture list should be 1 or len(files)')

    if len(args.skyap) != len(args.ap):
        raise ValueError('Length of sky aperture list should equal len(aper)')

    for aper,skyaper in zip(args.ap,args.skyap):
        if skyaper < aper:
            raise ValueError('Sky aperture must be larger than object aperture')
    
    if len(args.ap) == 1:
        for fname in args.files:
            profile(fname,args.ap,args.skyap)

    else:
        for fname,aper,skyaper in zip(args.files,args.ap,args.skyap):
            profile(fname,aper,skyaper)


            
        
    
    

    










if __name__ == '__main__':
    main()
