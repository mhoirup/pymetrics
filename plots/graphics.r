library(ggplot2)

# Source data
data = quantmod::getSymbols('CFR.SW', source = 'yahoo', auto.assign = F)

# Set overall plot theme
ptheme = theme_classic() +
    theme(text = element_text(family = "Palatino"), 
          axis.line.y = element_blank(),
          axis.line.x = element_line(size = .1),
          axis.text = element_text(size = 10), 
          plot.title = element_text(size = 12),
          plot.subtitle = element_text(size = 10),
          axis.ticks.y = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.background = element_rect(fill = 'white',colour = 'black'),
          legend.text = element_text(size = 10)
    )
theme_set(ptheme)

prices = na.omit(data[,ncol(data)])
names(prices) = 'price'

ggplot(prices, aes(Index, price)) +
    geom_hline(yintercept = seq(100, 800, 100), alpha = .05) +
    geom_line(aes(colour = 'Adjusted price at exchange close'), size = 0.5) +
    geom_line(aes(y = rollmean(price, 250, fill = NA, align = 'right'), 
                  colour ='Yearly moving average (250 days)')) +
    scale_colour_manual(values = c('black', 'tomato2')) +
    scale_y_continuous(labels = function(x) paste0('€', x)) +
    scale_x_date(date_breaks = "3 year", date_label = '%Y') +
    labs(x = '', y = '', title = 'RMS.PA Over Time') +
    theme(legend.position = c(.25, .75))

ggsave('~/pymetrics/plots/lineplot.png')

returns = na.omit(diff(log(prices)))
names(returns) = 'returns'
write.csv(returns, '~/pymetrics/hermes.csv')

ggplot(returns, aes(Index, returns)) +
    geom_hline(yintercept = seq(-.15, .15, .05), alpha = .05) +
    geom_line(size = 0.5) +
    scale_y_continuous(breaks = seq(-.1, .1, .10)) +
    scale_x_date(date_breaks = "3 year", date_label = '%Y') +
    labs(x='', y='', title = 'RMS.PA Logarithmic Returns')

ggsave('~/pymetrics/plots/returns.png')

moments = sapply(c(mean(returns), sd(returns)), round, 3)

ggplot(returns, aes(returns)) +
    geom_hline(yintercept = seq(0, 30, 5), alpha = .05) +
    geom_line(aes(y = ..density.., colour = 'Empirical'), stat = 'density',
              size = 1.2) +
    stat_function(fun = dnorm, args = list(mean(returns), sd(returns)), 
                  aes(colour = 'Gaussian'), size = 1.2) +
    geom_histogram(aes(y = stat(density)), binwidth = .005,
                   fill = 'white',colour = 'black', alpha = .1) +
    scale_colour_manual(values = c('black', 'tomato2')) +
    scale_x_continuous(breaks = seq(-.1, .1, .05)) +
    theme(legend.position = c(.14, .75)) +
    labs(y = '', x = '',
         title = 'Histogram of RMS.PA Returns with Overlaid Densities',
         subtitle = paste0('Gaussian density fitted with sample', 
                          ' moments; mu = ', moments[1], ', sigma = ',
                          moments[2]))

ggsave('~/pymetrics/plots/histogram.png')

ggplot(returns, aes(returns)) +
    geom_hline(yintercept = seq(0, 1, .125), alpha = .05) +
    stat_ecdf(aes(colour = 'Empirical'), geom = 'line', size = .7) +
    stat_function(fun = pnorm, args = list(mean(returns), sd(returns)), 
                  aes(colour = 'Gaussian'), size = .7) +
    scale_colour_manual(values = c('black', 'tomato2')) +
    scale_x_continuous(breaks = seq(-.1, .1, .05)) +
    theme(legend.position = c(.15, .75)) +
    labs(y = '', x = '',
         title = paste('Estimated Cumulative Distrubtion Function', 
                       'of RMS.PA Returns'),
         subtitle = paste0('Gaussian CDF fitted with sample', 
                          ' moments; mu = ', moments[1], ', sigma = ',
                          moments[2]))


ggsave('~/pymetrics/plots/ecdf.png')

m = 100
ci <- qnorm(.05/2) / sqrt(length(returns))
auto = acf(returns, m, plot = F)$acf[-1]
partial = pacf(returns, m, plot = F)$acf

setNames(data.frame(auto, partial), c('ACF', 'PACF')) %>%
    tidyr::gather(key = 'type', value = 'rho') %>%
    dplyr::mutate(lag = rep(seq(m), 2)) %>%
    ggplot(aes(lag, rho)) +
        geom_hline(yintercept = seq(-.06, .06, .02), alpha = .05) +
        geom_segment(mapping = aes(xend = lag, yend = 0), size = .3) +
        geom_hline(yintercept = c(ci, -ci), lty = 'dashed',
                   alpha = 1, colour = 'royalblue2') +
        geom_segment(aes(x = .5, xend = m + .5, y = 0, yend = 0), 
                     size = .15) +
        facet_wrap(~ type) +
        scale_y_continuous(breaks = seq(-.06, .06, .03)) +
        scale_x_continuous(breaks = c(1, 25, 50, 75, 100)) +
        labs(x = '', y = '',
             title = 'Sample Autocorrelations of RMS.PA Returns',
             subtitle = paste('Absolute confidence interval at',
                             round(abs(ci), 4))) +
        theme(strip.text.x = element_text(size = 10),
              strip.background = element_blank())

ggsave('~/pymetrics/plots/correlations.png')

