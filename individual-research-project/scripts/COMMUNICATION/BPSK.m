% This script is based on BPSK_Introduction.m
% EENG22000 Communications 2022 Lab Notes
% Communication Laboratory
% Author: Dr. Tommaso Cappello
% The author thanks Dr. Simon Armour
% Modified by Yumu Xie

% EbNo: signal/noise = S/N (SNR)
% When EbNo > 0, it means that signal is stronger than noise
% When EbNo < 0, it means that noise is stronger than signal

% BER = Ne/N
% Ne = BER * N
% The number of bit errors in the received sequence is determined by bit
% error rate and the total number of transmitted bits
% BER: bit error rate
% Ne: the number of bit errors in the received sequence
% N: the total number of transmitted bits

% Br: bit rate
% bit rate (bitrate or as a variable R) is the number of bits that are 
% conveyed or processed per unit of time
% bit/s

clc
close all
clearvars

% Script parameters:
Br = 1024;             % Bit rate
m = 1;                 % Bits per symbol (BPSK)
first_N = 20000000;    % Number of simulated bits
second_N = 5000000;    % Number of simulated bits
EbNo = -10;            % Eb/N0 in dB
OS = 8;                % Oversampling factor
rollOff = 0.5;         % Roll-off factor

% Design root raised cosine filter:
numSymbols = 15;                                 % Filter length in symbols
h = rcosdesign(rollOff, numSymbols, OS, 'sqrt'); % Generate coefficients
h = h./sum(h);

% Modulator:
first_txBit = rand(1,first_N)>0.5;                   % Generate bit sequence
first_txSymbol = first_txBit.*2 -1;                  % Convert to symbols
second_txBit = rand(1,second_N)>0.5;                 % Generate bit sequence
second_txSymbol = second_txBit.*2 -1;                % Convert to symbols

% Transmitter filter:
first_txSymbolUp = upsample(first_txSymbol,OS, 1);   % Upsample by zero stuffing
first_txSignal = conv(first_txSymbolUp, h);          % Filter to generate waveform
first_l = length(first_txSignal);                    % Calculate length
second_txSymbolUp = upsample(second_txSymbol,OS, 1); % Upsample by zero stuffing
second_txSignal = conv(second_txSymbolUp, h);        % Filter to generate waveform
second_l = length(second_txSignal);                  % Calculate length

% Channel:
first_PtxSignal = sum(first_txSignal.*conj(first_txSignal))/2;                         % Calculate power
first_Bs = Br/m;                                                                       % Symbol rate
first_Ns = first_N/m;                                                                  % Number of symbols
first_sigma = sqrt(first_PtxSignal*first_Bs/(2*first_Ns*Br)*10^(-EbNo/10));            % Calculate sigma
first_noise = randn(1,first_l).*first_sigma+1j.*randn(1,first_l).*first_sigma;         % Generate noise
first_rxSignal = first_txSignal + first_noise;                                         % AWGN channel model
second_PtxSignal = sum(second_txSignal.*conj(second_txSignal))/2;                      % Calculate power
second_Bs = Br/m;                                                                      % Symbol rate
second_Ns = second_N/m;                                                                % Number of symbols
second_sigma = sqrt(second_PtxSignal*second_Bs/(2*second_Ns*Br)*10^(-EbNo/10));        % Calculate sigma
second_noise = randn(1,second_l).*second_sigma+1j.*randn(1,second_l).*second_sigma;    % Generate noise
second_rxSignal = second_txSignal + second_noise;                                      % AWGN channel model

% Receiver filter:
first_rxSignalFilt = conv(first_rxSignal,h);                                   % Apply receiver filter
first_firDelay = numSymbols*OS + 1;                                            % FIR delay calculation
first_rxSymbol = first_rxSignalFilt(first_firDelay:OS:end-first_firDelay);     % Align and downsample
second_rxSignalFilt = conv(second_rxSignal,h);                                 % Apply receiver filter
second_firDelay = numSymbols*OS + 1;                                           % FIR delay calculation
second_rxSymbol = second_rxSignalFilt(second_firDelay:OS:end-second_firDelay); % Align and downsample

% Demodulate:
first_rxBit = real(first_rxSymbol)>0;                % Decode bits
second_rxBit = real(second_rxSymbol)>0;              % Decode bits

% Calculate BER:
first_numErrors = numel(find(first_txBit~=first_rxBit));    % Count the number of bit errors
first_BER = first_numErrors/first_N;                        % Simulated BER
first_BERtheory = berawgn(EbNo,'pam',2);                    % Theoretical BER (BPSK)
second_numErrors = numel(find(second_txBit~=second_rxBit)); % Count the number of bit errors
second_BER = second_numErrors/second_N;                     % Simulated BER
second_BERtheory = berawgn(EbNo,'pam',2);                   % Theoretical BER (BPSK)

disp(['EbNo: ', num2str(EbNo)]);
disp(['first numErrors: ', num2str(first_numErrors)]);
disp(['first BER: ', num2str(first_BER)]);
disp(['first BERtheory: ', num2str(first_BERtheory)]);
disp(['second numErrors: ', num2str(second_numErrors)]);
disp(['second BER: ', num2str(second_BER)]);
disp(['second BERtheory: ', num2str(second_BERtheory)]);
