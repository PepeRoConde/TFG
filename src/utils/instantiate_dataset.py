import os
from src.data.Online_Dataset import Online_Dataset
from src.data.Offline_Dataset import Offline_Dataset
from src.data.recorta_dataset import recorta_dataset
from src.data.RFMiD_Dataset import RFMiDDataset

def instantiate_dataset(args):

    if args.dataset == 'online':
        train_dataset = Online_Dataset(args.directorio_train_base, tamano_patch=args.tamano_patch, label_mode=args.label_mode, 
                                       sigma=args.sigma, num_sigmas=args.num_sigmas, aumento_datos=args.aumento_datos, 
                                       total_epochs=args.epochs, sobrelapamento=args.overlap_rate, contador_aumento=args.contador_aumento)

        val_dataset = Online_Dataset(args.directorio_val_base, tamano_patch=args.tamano_patch, label_mode=args.label_mode, 
                                     sigma=args.sigma, num_sigmas=args.num_sigmas, 
                                     total_epochs=args.epochs, sobrelapamento=args.overlap_rate, contador_aumento=args.contador_aumento)

    elif args.dataset == 'offline':
       
        # train -------

        directorio_train_cropeado = f'{args.directorio_train_base}_{args.tamano_patch}_{args.overlap_rate}'

        if not os.path.isdir(directorio_train_cropeado):
            print('no hay el conjunto de datos de patches que quieres para entrenar, esperte y te lo hago')
            recorta_dataset(input_dir=args.directorio_train_base, output_dir=directorio_train_cropeado, 
                               patch_size=args.tamano_patch,  overlap_rate=args.overlap_rate,
                               image_start_idx=21, image_end_idx=36)


        train_dataset = Offline_Dataset(directorio_train_cropeado,
                                        label_mode=args.label_mode, sigma=args.sigma, num_sigmas=args.num_sigmas,
                                      data_augmentation=args.aumento_datos, total_epochs=args.epochs)

        # val -----------

        directorio_val_cropeado = f'{args.directorio_val_base}_{args.tamano_patch}_{args.overlap_rate}'

        if not os.path.isdir(directorio_val_cropeado):
            print('no hay el conjunto de datos de patches que quieres para validar, esperte y te lo hago')
            recorta_dataset(input_dir=args.directorio_val_base, output_dir=directorio_val_cropeado, 
                               patch_size=args.tamano_patch,  overlap_rate=args.overlap_rate,
                               image_start_idx=37, image_end_idx=39)

        val_dataset = Offline_Dataset(directorio_val_cropeado,
                                        label_mode=args.label_mode, sigma=args.sigma, num_sigmas=args.num_sigmas,
                                      data_augmentation=False, total_epochs=args.epochs)
        
    elif args.dataset == 'rfmid':
        train_dataset = RFMiDDataset(
            data_dir=args.directorio_train_base,
            augmentation=args.aumento_datos,
            tamano_patch=args.tamano_patch,
            total_epochs=args.epochs
        )

        val_dataset = RFMiDDataset(
            data_dir=args.directorio_val_base,
            augmentation=False,
            tamano_patch=args.tamano_patch,
            total_epochs=args.epochs
        )

    else:
        raise NotImplementedError(f'La opcion {args.dataset} no existe, debe ser o "online" (cortar las imagenes bajo demanda) o "offline" (previamente se espera la ejecucion de src.data.crop_script)')

    return train_dataset, val_dataset
