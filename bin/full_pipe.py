









unlensed = get_unlensed()
lensed = downsample(get_lensed(unlensed))
save_lensed(lensed)



lensed = load_lensed()
beamed_clean = beam(lensed)
